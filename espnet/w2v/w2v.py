import json
import logging
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

from espnet.data.dataset import ASRDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.batchfy import make_batchset
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.utils.dynamic_import import dynamic_import
from espnet.asr.pytorch_backend.converter import CustomConverter
from espnet.asr.pytorch_backend.trainer import CustomTrainer
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.torch_utils import _recursive_to
from espnet.nets.pytorch_backend.nets_utils import pad_list


def build_model(list_input_dim, output_dim, options):

    if (options.enc_init is not None or options.dec_init is not None) and options.num_encs == 1:
        model = load_trained_modules(list_input_dim[0], output_dim, options)
    else:
        Model = dynamic_import(options.model_class)
        model = Model(
            list_input_dim[0] if options.num_encs == 1 else list_input_dim,
            output_dim,
            options
        )

    # write model config
    model_conf = options.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps(
            (list_input_dim[0] if options.num_encs == 1 else list_input_dim, output_dim, vars(options)),
            indent=4,
            ensure_ascii=False,
            sort_keys=True
        ).encode('utf_8'))

    for key in sorted(vars(options).keys()):
        logging.info('options: ' + key + ': ' + str(vars(options)[key]))

    for line in str(model).split("\n"):
        logging.info(line)

    return model


def get_optimizer(model, options):
    # Setup an optimizer
    if options.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=options.eps,
            weight_decay=options.weight_decay)
    elif options.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=options.weight_decay)
    elif options.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, options.adim, options.transformer_warmup_steps, options.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + options.opt)
    return optimizer


def setup(options, train=False):
    # SETUP AND PRELIMINARY OPERATIONS
    set_deterministic_pytorch(options)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)

    # check the use of multi-gpu
    if options.ngpu > 1:
        if options.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                options.batch_size, options.batch_size * options.ngpu))
            options.batch_size *= options.ngpu
        if options.num_encs > 1:
            # TODO(ruizhili): implement data parallel for multi-encoder setup.
            raise NotImplementedError("Data parallel is not supported for multi-encoder setup.")

    options.mtl_mode = 'att'
    options.use_apex = False

    # set torch device
    device = torch.device("cuda" if options.ngpu > 0 else "cpu")
    if options.train_dtype not in ("float16", "float32", "float64"):
        options.train_dtype = "float32"
    dtype = getattr(torch, options.train_dtype)

    json_path = options.valid_json if train else options.test_json

    validation_set = ASRDataset.from_json(json_path)
    input_dims = validation_set.input_dim
    output_dim = validation_set.output_dim[0]

    for i in range(options.num_encs):
        logging.info('stream{}: input dims : {}'.format(i + 1, input_dims[i]))
    logging.info('#output dims: ' + str(output_dim))

    options.use_sortagrad = (options.sortagrad == -1 or options.sortagrad > 0)

    # MODEL AND OPTIMIZER CONFIGURATION
    model = build_model(input_dims, output_dim, options)
    if not train:
        model.eval()
        
    reporter = model.reporter
    model = model.to(device=device, dtype=dtype)
    optimizer = get_optimizer(model, options)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)

    return model, optimizer, converter, device


def load_dataset(json_path, options, converter, train=True):
    with open(json_path, "rb") as json_file:
        metadata = json.load(json_file)["utts"]

    dataset = make_batchset(
        metadata, options.batch_size,
        options.maxlen_in, options.maxlen_out, options.minibatches,
        min_batch_size=options.ngpu if options.ngpu > 1 else 1,
        shortest_first=options.use_sortagrad if train else False,
        count=options.batch_count,
        batch_bins=options.batch_bins,
        batch_frames_in=options.batch_frames_in,
        batch_frames_out=options.batch_frames_out,
        batch_frames_inout=options.batch_frames_inout,
        iaxis=0, oaxis=0
    )

    loader = LoadInputsAndTargets(
        mode='asr',
        load_output=True,
        preprocess_conf=options.preprocess_conf,
        preprocess_args={'train': train}  # Switch the mode of preprocessing
    )

    iterator = ChainerDataLoader(
        TransformDataset(dataset, lambda data: converter([loader(data)])),
        batch_size=1,
        shuffle=not(options.use_sortagrad) if train else False,
        num_workers=options.n_iter_processes,
        collate_fn=lambda x: x[0]
    )

    return iterator, dataset, loader, metadata


def train(options):
    """Train with the given args.

    Args:reporter
        options (namespace): The program arguments.

    """

    model, optimizer, converter, device = setup(options)

    # DATA ITERATORS
    train_iter, *_ = load_dataset(options.train_json, options, converter, train=True)
    valid_iter, _, load_valid, valid_json = load_dataset(options.valid_json, options, converter, train=False)

    # TRAINER CONFIGURATION
    # Set up a trainer
    trainer = CustomTrainer(
        options, model, optimizer, train_iter, valid_iter, converter, device, valid_json, load_valid
    )

    logging.info("Start training")
    trainer.run()


def recog(options):
    """Train with the given args.

    Args:reporter
        options (namespace): The program arguments.

    """

    # SETUP AND PRELIMINARY OPERATIONS
    model, optimizer, converter, device = setup(options, train=False)

    testing_set = ASRDataset.from_json(options.test_json, sort_by_length=True)

    test_iter = ChainerDataLoader(
        testing_set,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=testing_set.collate_samples
    )

    decoder = model.decoder
    clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(
        decoder.embed.weight.cpu(), np.arange(len(decoder.char_list))
    )

    # FIXME DIRTY HACK:
    # Everything is kinda repeated and should be implemented / optimised in the model methods
    with torch.no_grad():
        start = 0
        for i, batch in enumerate(test_iter):
            batch = _recursive_to(batch, device)
            X, Xlens = batch["input1"], batch["input1_length"]
            y, ylens = batch["target1"], batch["target1_length"]
            uttids = batch["uttid"]
            batch_size = len(X)
            end = start + batch_size

            sentences = [y[y != decoder.ignore_id] for y in y]
            prev_output_tokens = decoder._add_sos_token(sentences, pad=True, pad_value=decoder.eos)

            eos_tokens = torch.ones_like(y[:, :1]) * decoder.eos
            target = torch.cat([y.masked_fill(y == -1, decoder.eos), eos_tokens], -1)

            for tensor_name, tensor in zip(("X", "Xlens", "y", "prev_output_tokens"), (X, Xlens, y, prev_output_tokens)):
                logging.info(f"{tensor_name}: {tensor.type()} {tensor.size()}")

            y_pred, att = model.evaluate(X, Xlens, prev_output_tokens)
            att = torch.stack(att).transpose(0, 1).detach().cpu().numpy()

            target_mask = (target == decoder.eos)
            target_emb = decoder.embed(target)

            losses = F.mse_loss(target_emb, y_pred, reduction="none").mean(-1)
            loss = (losses.masked_fill(target_mask, 0).sum(-1) / (ylens + 1)).mean()

            logging.info(f"Loss: {loss:.4f}")

            predictions = y_pred.detach().cpu().numpy()

            target_sentence = decoder.tokens_to_string(target[-1])
            predicted_sentence = decoder.tokens_to_string(clf.predict(predictions[-1]))

            logging.info(f"Target: {target_sentence}")
            logging.info(f"Prediction: {predicted_sentence}")

            import ipdb; ipdb.set_trace()

            # predicted_tokens = clf.predict(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape[:-1])
            # accuracy = (predicted_tokens == target).mean()

            target_emb = target_emb.detach().cpu().numpy()

            np.save(f"{options.outdir}/id_batch{i}.npy", uttids)
            np.save(f"{options.outdir}/att_weights_batch{i}.npy", att)
            np.save(f"{options.outdir}/predictions_batch{i}.npy", predictions)
            np.save(f"{options.outdir}/target_batch{i}.npy", target_emb)
            np.save(f"{options.outdir}/target_lengths{i}.npy", ylens.cpu().detach().numpy())

            start = end
