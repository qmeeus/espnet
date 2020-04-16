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
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.torch_utils import _recursive_to


def build_model(list_input_dim, output_dim, args):

    if (args.enc_init is not None or args.dec_init is not None) and args.num_encs == 1:
        model = load_trained_modules(list_input_dim[0], output_dim, args)
    else:
        model_class = dynamic_import(args.model_class)
        model = model_class(
            list_input_dim[0] if args.num_encs == 1 else list_input_dim,
            output_dim,
            args
        )

    # write model config
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps(
            (list_input_dim[0] if args.num_encs == 1 else list_input_dim, output_dim, vars(args)),
            indent=4,
            ensure_ascii=False,
            sort_keys=True
        ).encode('utf_8'))

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    for line in str(model).split("\n"):
        logging.info(line)

    return model


def get_optimizer(model, args):
    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)
    return optimizer


def recog(args):
    """Train with the given args.

    Args:reporter
        args (namespace): The program arguments.

    """
    # SETUP AND PRELIMINARY OPERATIONS
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    testing_set = ASRDataset.from_json(args.test_json)
    test_itr = DataLoader(
        testing_set, 
        batch_size=args.batch_size,
        collate_fn=testing_set.collate_samples
    )

    input_dim_list = idim_list = testing_set.input_dim
    output_dim = odim = testing_set.output_dim[0]

    for i in range(args.num_encs):
        logging.info('stream{}: input dims : {}'.format(i + 1, idim_list[i]))
    logging.info('#output dims: ' + str(odim))

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu
        if args.num_encs > 1:
            # TODO(ruizhili): implement data parallel for multi-encoder setup.
            raise NotImplementedError("Data parallel is not supported for multi-encoder setup.")

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        args.mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        args.mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        args.mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # MODEL AND OPTIMIZER CONFIGURATION
    model = build_model(idim_list, odim, args)
    for line in str(model).split("\n"):
        logging.info(line)
        
    reporter = model.reporter
    model = model.to(device=device, dtype=dtype)
    model.eval()
    optimizer = get_optimizer(model, args)

    args.use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # DATA ITERATORS
    # Setup a converter
    converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)

    # read json data
    with open(args.test_json, 'rb') as f:
        test_json = json.load(f)['utts']

    args.use_sortagrad = (args.sortagrad == -1 or args.sortagrad > 0)
    # make minibatch list (variable length)

    # test = make_batchset(
    #     test_json, args.batch_size,
    #     args.maxlen_in, args.maxlen_out, args.minibatches,
    #     min_batch_size=args.ngpu if args.ngpu > 1 else 1,
    #     shortest_first=False,
    #     count=args.batch_count,
    #     batch_bins=args.batch_bins,
    #     batch_frames_in=args.batch_frames_in,
    #     batch_frames_out=args.batch_frames_out,
    #     batch_frames_inout=args.batch_frames_inout,
    #     iaxis=0, oaxis=0
    # )

    # load_test = LoadInputsAndTargets(
    #     mode='asr',
    #     load_output=True,
    #     preprocess_conf=args.preprocess_conf,
    #     preprocess_args={'train': False}
    # )

    test_iter = ChainerDataLoader(
        testing_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=testing_set.collate_samples
    )

    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    # test_iter = {
    #     'main': ChainerDataLoader(
    #         TransformDataset(test, lambda data: converter([load_test(data)])),
    #         batch_size=1,
    #         shuffle=not(args.use_sortagrad) if i == 0 else False,
    #         num_workers=args.n_iter_processes,
    #         collate_fn=lambda x: x[0]
    #     )
    # }


    decoder = model.decoder
    clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(
        decoder.embed.weight.cpu(), np.arange(len(decoder.char_list))
    )

    # FIXME DIRTY HACK:
    # Everything is kinda repeated and should be implemented / optimised in the model methods
    with torch.no_grad():

        ids = np.array(list(test_json))
        start = 0
        for i, batch in enumerate(test_iter):
            X, Xlens, y, ylens = _recursive_to(batch, device)
            
            batch_size = len(X)
            end = start + batch_size

            sentences = [y[y != decoder.ignore_id] for y in y]
            prev_output_tokens = decoder._add_sos_token(sentences, pad=True, pad_value=decoder.eos)

            eos_tokens = torch.ones_like(y[:, :1]) * decoder.eos
            target = torch.cat([y.masked_fill(y == -1, decoder.eos), eos_tokens], -1)

            for tensor_name, tensor in zip(("X", "Xlens", "y", "prev_output_tokens"), (X, Xlens, y, prev_output_tokens)):
                logging.info(f"{tensor_name}: {tensor.type()} {tensor.size()}")

            import ipdb; ipdb.set_trace()
            y_pred, att = model.evaluate(X, Xlens, prev_output_tokens)
            att = torch.stack(att).transpose(0, 1).detach().cpu().numpy()

            target_mask = (target == decoder.eos)
            target_emb = decoder.embed(target) 

            losses = F.mse_loss(target_emb, y_pred, reduction="none").mean(-1)
            loss = (losses.masked_fill(target_mask, 0).sum(-1) / ylens).mean()

            predictions = y_pred.detach().cpu().numpy()
            predicted_tokens = clf.predict(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape[:-1])
            accuracy = (predicted_tokens == target).mean()

            target_emb = target_emb.detach().cpu().numpy()

            np.save(f"{args.outdir}/id_batch{i}.npy", ids[start:end])
            np.save(f"{args.outdir}/att_weights_batch{i}.npy", att)
            np.save(f"{args.outdir}/predictions_batch{i}.npy", predictions)
            np.save(f"{args.outdir}/target_batch{i}.npy", target_emb)
            np.save(f"{args.outdir}/target_lengths{i}.npy", ylens.cpu().detach().numpy())

            start = end
