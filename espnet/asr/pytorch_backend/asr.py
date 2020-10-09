#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""

import json
import logging
import os
import sys

import numpy as np
import torch

from torch.utils.data import DataLoader

from espnet.data.dataset import ASRDataset
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.asr.pytorch_backend.converter import CustomConverter
from espnet.asr.pytorch_backend.trainer import CustomTrainer

from espnet.nets.asr_interface import ASRInterface
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest

DEBUG_MODEL = False


def count_parameters(model):
    n_params, n_trainable = 0, 0
    for p in model.parameters():
        np = p.numel()
        n_params += np
        if p.requires_grad:
            n_trainable += np
    return n_params, n_trainable

def build_model(list_input_dim, output_dim, args):

    if (args.enc_init is not None or args.dec_init is not None) and args.num_encs == 1:
        model = load_trained_modules(list_input_dim[0], output_dim, args)
    else:
        Model = dynamic_import(args.model_class)
        model = Model(
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

    return model

def display_model(model):
    for line in str(model).split("\n"):
        logging.info(line)

    n_params, n_trainable = count_parameters(model)
    logging.info(f"# Params: {n_params / 10**6:.1f}M (trainable: {n_trainable / 10**6:.1f}M)")


def get_optimizer(model, args, reporter):
    # Setup an optimizer
    if args.opt == 'adadelta':
        
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay
        )

    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            weight_decay=args.weight_decay
        )

    elif args.opt == 'noam':

        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)

    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    return optimizer


def train(args):
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

    if not DEBUG_MODEL:
        training_set, validation_set = (
            ASRDataset.from_json(getattr(args, json_file))
            for json_file in ('train_json', 'valid_json')
        )

        train_itr, valid_itr = (
            DataLoader(dataset, batch_size=args.batch_size,
                    collate_fn=dataset.collate_samples)
            for dataset in (training_set, validation_set)
        )

        input_dim_list = idim_list = training_set.input_dim
        output_dim = odim = training_set.output_dim[0]

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
    if args.freeze_encoder == -1:
        args.mtl_mode = 'mtl'
        args.mtlalpha = 0
        args.alpha_scheduler = None
    elif getattr(args, "alpha_scheduler", None):
        args.mtl_mode = 'mtl'
        args.mtlalpha = args.alpha_scheduler[0]
        logging.info("Multitask learning mode with scheduler")
    elif args.mtlalpha == 1.0:
        args.mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        args.mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        args.mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    if DEBUG_MODEL:
        from pytorch_lightning import Trainer
        from espnet.asr.pytorch_backend.model import ASRModel

        model_class = dynamic_import(args.model_class)
        model = ASRModel(model_class, args)
        trainer = Trainer(gpus=[1], log_gpu_memory='all', overfit_pct=.01, profiler=True)
        trainer.fit(model)
        return

    # MODEL AND OPTIMIZER CONFIGURATION
    assert type(args.char_list) == list
    model = build_model(idim_list, len(args.char_list), args)
        
    reporter = model.reporter
    model = model.to(device=device, dtype=dtype)

    if args.freeze_encoder == -1:
        logging.warn("Freeze the encoder and disabling gradients for this module")
        optimizer = get_optimizer(model.decoder, args, reporter)
        for module in args.enc_init_mods:
            for param in getattr(model, module.strip(".")).parameters():
                param.requires_grad = False
    else:
        optimizer = get_optimizer(model, args, reporter)
        if args.freeze_encoder > 0:
            decoder_optimizer = get_optimizer(model.decoder, args, reporter)

    display_model(model)

    args.use_apex = False

    # DATA ITERATORS
    # Setup a converter
    converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    args.use_sortagrad = (args.sortagrad == -1 or args.sortagrad > 0)
    # make minibatch list (variable length)

    train, valid = (
        make_batchset(
            json_file, args.batch_size,
            args.maxlen_in, args.maxlen_out, args.minibatches,
            min_batch_size=args.ngpu if args.ngpu > 1 else 1,
            shortest_first=args.use_sortagrad if i == 0 else False,
            count=args.batch_count,
            batch_bins=args.batch_bins,
            batch_frames_in=args.batch_frames_in,
            batch_frames_out=args.batch_frames_out,
            batch_frames_inout=args.batch_frames_inout,
            iaxis=0, oaxis=0
        ) for i, json_file in enumerate([train_json, valid_json])
    )

    load_tr, load_cv = (
        LoadInputsAndTargets(
            mode='asr',
            load_output=True,
            preprocess_conf=args.preprocess_conf,
            preprocess_args={'train': is_train}  # Switch the mode of preprocessing
        ) for is_train in (True, False)
    )

    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter, valid_iter = (
        {
            'main': ChainerDataLoader(
                dataset=TransformDataset(subset, lambda data: converter([loader(data)])),
                batch_size=1,
                shuffle=not(args.use_sortagrad) if i == 0 else False,
                num_workers=args.n_iter_processes,
                collate_fn=lambda x: x[0]
            )
        } for i, (subset, loader) in enumerate(zip([train, valid], [load_tr, load_cv]))
    )

    # TRAINER CONFIGURATION
    # Set up a trainer
    if args.freeze_encoder > 0:
        optimizer = {"main": optimizer, "decoder": decoder_optimizer}

    trainer = CustomTrainer(
        args, model, optimizer, train_iter, valid_iter, converter, device, valid_json, load_cv
    )

    logging.info("Start training")
    trainer.run()

def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    rnnlm = None

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False}
    )

    device = "cuda" if args.ngpu > 0 else "cpu"

    if args.batchsize < 2:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = feat[0][0] if args.num_encs == 1 else [feat[idx][0] for idx in range(model.num_encs)]
                feat = torch.as_tensor(feat).to(device)
                if hasattr(model, "decoder_mode") and model.decoder_mode == "maskctc":
                    nbest_hyps = model.recognize_maskctc(
                        feat, args, train_args.char_list
                    )
                else:
                    nbest_hyps = model.recognize(feat, args, train_args.char_list, rnnlm)
                new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    else:
        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feat = load_inputs_and_targets(batch)[0] if args.num_encs == 1 else load_inputs_and_targets(batch)
                feat = torch.as_tensor(feat).to(device)
                nbest_hyps = model.recognize_batch(torch.as_tensor(feat), args, train_args.char_list, rnnlm=rnnlm)

                for i, nbest_hyp in enumerate(nbest_hyps):
                    name = names[i]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))


def enhance(args):
    """Dumping enhanced speech and mask.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # TODO(ruizhili): implement enhance for multi-encoder model
    assert args.num_encs == 1, "number of encoder should be 1 ({} is given)".format(args.num_encs)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    model_class = dynamic_import(train_args.model_class)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, ASRInterface)
    torch_load(args.model, model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=None  # Apply pre_process in outer func
    )
    if args.batchsize == 0:
        args.batchsize = 1

    # Creates writers for outputs from the network
    if args.enh_wspecifier is not None:
        enh_writer = file_writer_helper(args.enh_wspecifier,
                                        filetype=args.enh_filetype)
    else:
        enh_writer = None

    # Creates a Transformation instance
    preprocess_conf = (
        train_args.preprocess_conf if args.preprocess_conf is None
        else args.preprocess_conf)
    if preprocess_conf is not None:
        logging.info('Use preprocessing'.format(preprocess_conf))
        transform = Transformation(preprocess_conf)
    else:
        transform = None

    # Creates a IStft instance
    istft = None
    frame_shift = args.istft_n_shift  # Used for plot the spectrogram
    if args.apply_istft:
        if preprocess_conf is not None:
            # Read the conffile and find stft setting
            with open(preprocess_conf) as f:
                # Json format: e.g.
                #    {"process": [{"type": "stft",
                #                  "win_length": 400,
                #                  "n_fft": 512, "n_shift": 160,
                #                  "window": "han"},
                #                 {"type": "foo", ...}, ...]}
                conf = json.load(f)
                assert 'process' in conf, conf
                # Find stft setting
                for p in conf['process']:
                    if p['type'] == 'stft':
                        istft = IStft(win_length=p['win_length'],
                                      n_shift=p['n_shift'],
                                      window=p.get('window', 'hann'))
                        logging.info('stft is found in {}. '
                                     'Setting istft config from it\n{}'
                                     .format(preprocess_conf, istft))
                        frame_shift = p['n_shift']
                        break
        if istft is None:
            # Set from command line arguments
            istft = IStft(win_length=args.istft_win_length,
                          n_shift=args.istft_n_shift,
                          window=args.istft_window)
            logging.info('Setting istft config from the command line args\n{}'
                         .format(istft))

    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return zip_longest(*kargs, fillvalue=fillvalue)

    num_images = 0
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for names in grouper(args.batchsize, keys, None):
        batch = [(name, js[name]) for name in names]

        # May be in time region: (Batch, [Time, Channel])
        org_feats = load_inputs_and_targets(batch)[0]
        if transform is not None:
            # May be in time-freq region: : (Batch, [Time, Channel, Freq])
            feats = transform(org_feats, train=False)
        else:
            feats = org_feats

        with torch.no_grad():
            enhanced, mask, ilens = model.enhance(feats)

        for idx, name in enumerate(names):
            # Assuming mask, feats : [Batch, Time, Channel. Freq]
            #          enhanced    : [Batch, Time, Freq]
            enh = enhanced[idx][:ilens[idx]]
            mas = mask[idx][:ilens[idx]]
            feat = feats[idx]

            # Plot spectrogram
            if args.image_dir is not None and num_images < args.num_images:
                import matplotlib.pyplot as plt
                num_images += 1
                ref_ch = 0

                plt.figure(figsize=(20, 10))
                plt.subplot(4, 1, 1)
                plt.title('Mask [ref={}ch]'.format(ref_ch))
                plot_spectrogram(plt, mas[:, ref_ch].T, fs=args.fs,
                                 mode='linear', frame_shift=frame_shift,
                                 bottom=False, labelbottom=False)

                plt.subplot(4, 1, 2)
                plt.title('Noisy speech [ref={}ch]'.format(ref_ch))
                plot_spectrogram(plt, feat[:, ref_ch].T, fs=args.fs,
                                 mode='db', frame_shift=frame_shift,
                                 bottom=False, labelbottom=False)

                plt.subplot(4, 1, 3)
                plt.title('Masked speech [ref={}ch]'.format(ref_ch))
                plot_spectrogram(
                    plt, (feat[:, ref_ch] * mas[:, ref_ch]).T,
                    frame_shift=frame_shift,
                    fs=args.fs, mode='db', bottom=False, labelbottom=False)

                plt.subplot(4, 1, 4)
                plt.title('Enhanced speech')
                plot_spectrogram(plt, enh.T, fs=args.fs,
                                 mode='db', frame_shift=frame_shift)

                plt.savefig(os.path.join(args.image_dir, name + '.png'))
                plt.clf()

            # Write enhanced wave files
            if enh_writer is not None:
                if istft is not None:
                    enh = istft(enh)
                else:
                    enh = enh

                if args.keep_length:
                    if len(org_feats[idx]) < len(enh):
                        # Truncate the frames added by stft padding
                        enh = enh[:len(org_feats[idx])]
                    elif len(org_feats) > len(enh):
                        padwidth = [(0, (len(org_feats[idx]) - len(enh)))] \
                            + [(0, 0)] * (enh.ndim - 1)
                        enh = np.pad(enh, padwidth, mode='constant')

                if args.enh_filetype in ('sound', 'sound.hdf5'):
                    enh_writer[name] = (args.fs, enh)
                else:
                    # Hint: To dump stft_signal, mask or etc,
                    # enh_filetype='hdf5' might be convenient.
                    enh_writer[name] = enh

            if num_images >= args.num_images and enh_writer is None:
                logging.info('Breaking the process.')
                break
