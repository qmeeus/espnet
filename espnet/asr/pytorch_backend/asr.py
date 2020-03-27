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
from espnet.asr.asr_utils import format_mulenc_args
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.asr.pytorch_backend.converter import CustomConverter, CustomConverterMulEnc
from espnet.asr.pytorch_backend.trainer import CustomTrainer

import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
# from espnet.transform.spectrogram import IStft
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

def build_model(list_input_dim, output_dim, args):

    if (args.enc_init is not None or args.dec_init is not None) and args.num_encs == 1:
        model = load_trained_modules(list_input_dim[0], output_dim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(
            list_input_dim[0] if args.num_encs == 1 else list_input_dim,
            output_dim,
            args
        )

    assert isinstance(model, ASRInterface)

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

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
    # TODO: use functools.partial and return partially implemented optimizer
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

def train(args):
    """Train with the given args.

    Args:reporter
        args (namespace): The program arguments.

    """
    # SETUP AND PRELIMINARY OPERATIONS
    set_deterministic_pytorch(args)
    if args.num_encs > 1:
        args = format_mulenc_args(args)

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
    if args.mtlalpha == 1.0:
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
        import ipdb; ipdb.set_trace()

        model_class = dynamic_import(args.model_module)
        model = ASRModel(model_class, args)
        trainer = Trainer(gpus=[1], log_gpu_memory='all', overfit_pct=.01, profiler=True)
        trainer.fit(model)
        return

    # MODEL AND OPTIMIZER CONFIGURATION
    model = build_model(idim_list, odim, args)
    reporter = model.reporter
    model = model.to(device=device, dtype=dtype)
    optimizer = get_optimizer(model, args)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(f"You need to install apex for --train-dtype {args.train_dtype}. "
                          "See https://github.com/NVIDIA/apex#linux")
            raise e
        if args.opt == 'noam':
            model, optimizer.optimizer = amp.initialize(model, optimizer.optimizer, opt_level=args.train_dtype)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.train_dtype)
        args.use_apex = True
    else:
        args.use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # DATA ITERATORS
    # Setup a converter
    if args.num_encs == 1:
        converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)
    else:
        converter = CustomConverterMulEnc([i[0] for i in model.subsample_list], dtype=dtype)

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
                dataset = TransformDataset(subset, lambda data: converter([loader(data)])),
                batch_size=1,
                shuffle=not(args.use_sortagrad) if i == 0 else False,
                num_workers=args.n_iter_processes,
                collate_fn=lambda x: x[0]
            )
        } for i, (subset, loader) in enumerate(zip([train, valid], [load_tr, load_cv]))
    )

    # TRAINER CONFIGURATION
    # Set up a trainer
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

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError("use '--api v2' option to decode with non-default language model")
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit, rnnlm_args.embed_unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit, rnnlm_args.embed_unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:

            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(
                    word_rnnlm.predictor,
                    rnnlm.predictor,
                    word_dict,
                    char_dict
                )
            )

        else:

            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(
                    word_rnnlm.predictor,
                    word_dict,
                    char_dict
                )
            )

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
        preprocess_args={'train': False})

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)
                feat = feat[0][0] if args.num_encs == 1 else [feat[idx][0] for idx in range(model.num_encs)]
                if args.streaming_mode == 'window' and args.num_encs == 1:
                    logging.info('Using streaming recognizer with window size %d frames', args.streaming_window)
                    se2e = WindowStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    for i in range(0, feat.shape[0], args.streaming_window):
                        logging.info('Feeding frames %d - %d', i, i + args.streaming_window)
                        se2e.accept_input(feat[i:i + args.streaming_window])
                    logging.info('Running offline attention decoder')
                    se2e.decode_with_attention_offline()
                    logging.info('Offline attention decoder finished')
                    nbest_hyps = se2e.retrieve_recognition()
                elif args.streaming_mode == 'segment' and args.num_encs == 1:
                    logging.info('Using streaming recognizer with threshold value %d', args.streaming_min_blank_dur)
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({'yseq': [], 'score': 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i:i + r])
                        if hyps is not None:
                            text = ' '.join([train_args.char_list[int(x)]
                                            for x in hyps[0]['yseq'][1:-1] if int(x) != -1])
                            text = text.replace('\u2581', ' ').strip()  # for SentencePiece
                            text = text.replace('▁', ' ')
                            text = text.replace(model.space, ' ')
                            text = text.replace(model.blank, '')
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]['yseq'].extend(hyps[n]['yseq'])
                                nbest_hyps[n]['score'] += hyps[n]['score']
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
                feats = load_inputs_and_targets(batch)[0] if args.num_encs == 1 else load_inputs_and_targets(batch)
                if args.streaming_mode == 'window' and args.num_encs == 1:
                    raise NotImplementedError
                elif args.streaming_mode == 'segment' and args.num_encs == 1:
                    if args.batchsize > 1:
                        raise NotImplementedError
                    feat = feats[0]
                    nbest_hyps = []
                    for n in range(args.nbest):
                        nbest_hyps.append({'yseq': [], 'score': 0.0})
                    se2e = SegmentStreamingE2E(e2e=model, recog_args=args, rnnlm=rnnlm)
                    r = np.prod(model.subsample)
                    for i in range(0, feat.shape[0], r):
                        hyps = se2e.accept_input(feat[i:i + r])
                        if hyps is not None:
                            text = ' '.join([train_args.char_list[int(x)]
                                            for x in hyps[0]['yseq'][1:-1] if int(x) != -1])
                            text = text.replace('\u2581', ' ').strip()  # for SentencePiece
                            text = text.replace('▁', ' ')
                            text = text.replace(model.space, ' ')
                            text = text.replace(model.blank, '')
                            logging.info(text)
                            for n in range(args.nbest):
                                nbest_hyps[n]['yseq'].extend(hyps[n]['yseq'])
                                nbest_hyps[n]['score'] += hyps[n]['score']
                    nbest_hyps = [nbest_hyps]
                else:
                    nbest_hyps = model.recognize_batch(feats, args, train_args.char_list, rnnlm=rnnlm)

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
    model_class = dynamic_import(train_args.model_module)
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
