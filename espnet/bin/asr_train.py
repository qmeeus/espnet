#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Automatic speech recognition model training script."""

import logging
import os
import random
import subprocess
import sys
import re

import configargparse
import numpy as np
import torch
from espnet.utils.cli_utils import strtobool, count_gpus
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import load_dictionary
from distutils.util import strtobool as strtobool_

# TODO: compat --> replace with flags / options (= 4: rnn-type, vgg-frontend, bidirectional-encoder, pyramidal)
AVAILABLE_E2E_RNNS = [
    f"{front}{b}{cell}{p}"
    for front in ('vgg', '')
    for b in ('b', '')
    for cell in ('lstm', 'gru')
    for p in ('p', '')
]


def int_or_list_of_ints(string):
    if re.match(r"\[\d(?:\,\d)*\]", string):
        return [int(i) for i in eval(string)]
    return int(string)

def strtobool(string):
    # distutils.util.strtobool returns integer, but it's confusing,
    return bool(strtobool_(string))

def split_string(sep):

    def _strtolist(string):
        return list(map(str.strip, string.split(sep))) if string else []

    return _strtolist


CONFIG = {

    # General
    "gpus": dict(default=1, type=int_or_list_of_ints, help="# GPUS (for pytorch_lightning trainer)"),
    "ngpu": dict(default=None, type=int, help='Number of GPUs. If not given, use all visible devices'),
    "v1": dict(action="store_true"),
    "train-dtype": dict(
        default="float32",
        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
        help='Data type for training (only pytorch backend). O0,O1,.. flags require apex. '
        'See https://nvidia.github.io/apex/amp.html#opt-levels'
    ),
    "outdir": dict(type=str, required=True, help='Output directory'),
    "debugmode": dict(default=1, type=int, help='Debugmode'),
    "dict": dict(required=True, help='Dictionary'),
    "seed": dict(default=1, type=int, help='Random seed'),
    "debugdir": dict(type=str, help='Output directory for debugging'),
    "resume": dict(default='', nargs='?', help='Resume the training from snapshot'),
    "minibatches": dict(type=int, default='-1', help='Process only N minibatches (for debug)'),
    "verbose": dict(default=0, type=int, help='Verbose option'),
    "tensorboard-dir": dict(default=None, type=str, nargs='?', help="Tensorboard log dir path"),
    "report-interval-iters": dict(default=100, type=int, help="Report interval iterations"),
    "save-interval-iters": dict(default=0, type=int, help="Save snapshot interval iterations"),

    # Task
    "task": dict(type=str, default="asr_v1", help="Prediction task"),
    "train-json": dict(type=str, default=None, help='Filename of train label data (json)'),
    "valid-json": dict(type=str, default=None, help='Filename of validation label data (json)'),
    "model-class": dict(type=str, default='espnet.nets.pytorch_backend.e2e_asr:E2E', help='Python module, where to find the model'),
    "num-spkrs": dict(default=1, type=int, choices=[1, 2], help='Number of speakers in the speech.'),
    "backend": dict(default="pytorch", choices=["pytorch"]),

    # Recognition
    "report-cer": dict(default=False, action='store_true', help='Compute CER on development set'),
    "report-wer": dict(default=False, action='store_true', help='Compute WER on development set'),
    "nbest": dict(type=int, default=1, help='Output N-best hypotheses'),
    "beam-size": dict(type=int, default=4, help='Beam size'),
    "penalty": dict(default=0.0, type=float, help='Incertion penalty'),
    "maxlenratio": dict(
        default=0.0, type=float,
        help="Input length ratio to obtain max output length.\n"
        "If maxlenratio=0.0 (default), it uses a end-detect function to automatically find maximum hypothesis lengths"),
    "minlenratio": dict(default=0.0, type=float, help='Input length ratio to obtain min output length'),
    "ctc-weight": dict(default=0.3, type=float, help='CTC weight in joint decoding'),
    "rnnlm": dict(type=str, default=None, help='RNNLM model file to read'),
    "rnnlm-conf": dict(type=str, default=None, help='RNNLM model config file to read'),
    "lm-weight": dict(default=0.1, type=float, help='RNNLM weight.'),
    "sym-space": dict(default='<space>', type=str, help='Space symbol'),
    "sym-blank": dict(default='<blank>', type=str, help='Blank symbol'),

    # Batches
    "sortagrad": dict(default=0, type=int, nargs='?', help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs"),
    "batch-count": dict(
        default='auto', choices=BATCH_COUNT_CHOICES,
        help='How to count batch_size. The default (auto) will find how to count by args.'),
    "batch-size": dict(default=0, type=int, help='Maximum seqs in a minibatch (0 to disable)'),
    "batch-bins": dict(default=0, type=int, help='Maximum bins in a minibatch (0 to disable)'),
    "batch-frames-in": dict(default=0, type=int, help='Maximum input frames in a minibatch (0 to disable)'),
    "batch-frames-out": dict(default=0, type=int, help='Maximum output frames in a minibatch (0 to disable)'),
    "batch-frames-inout": dict(
        default=0, type=int, help='Maximum input+output frames in a minibatch (0 to disable)'),
    "maxlen-in": dict(
        default=800, type=int, metavar='ML',
        help='When --batch-count=seq, batch size is reduced if the input sequence length > ML.'),
    "maxlen-out": dict(
        default=150, type=int, metavar='ML',
        help='When --batch-count=seq, batch size is reduced if the output sequence length > ML'),
    "n-iter-processes": dict(default=0, type=int, help='Number of processes of iterator'),
    "preprocess-conf": dict(type=str, default=None, nargs='?', help='The configuration file for the pre-processing'),

    # Optimisation
    "opt": dict(default='adadelta', type=str, choices=['adadelta', 'adam', 'noam'], help='Optimizer'),
    "accum-grad": dict(default=1, type=int, help='Number of gradient accumuration'),
    "eps": dict(default=1e-8, type=float, help='Epsilon constant for optimizer'),
    "eps-decay": dict(default=0.01, type=float, help='Decaying ratio of epsilon'),
    "weight-decay": dict(default=0.0, type=float, help='Weight decay ratio'),
    "criterion": dict(default='loss', type=str, choices=['loss', 'accuracy'], help='Criterion to perform epsilon decay'),
    "threshold": dict(default=1e-4, type=float, help='Threshold to stop iteration'),
    "epochs": dict(default=30, type=int, help='Maximum number of epochs'),
    "early-stop-criterion": dict(
        default='validation/main/loss', type=str, nargs='?',
        help="Value to monitor to trigger an early stopping of the training"),
    "patience": dict(default=3, type=int, nargs='?', help="Number of epochs to wait without improvement before stopping the training"),
    "grad-clip": dict(default=5, type=float, help='Gradient norm threshold to clip'),
    "num-save-attention": dict(default=3, type=int, help='Number of samples of attention to be saved'),
    "grad-noise": dict(type=strtobool, default=False, help='The flag to switch to use noise injection to gradients during training'),

    # Pretrained models
    "enc-init": dict(default=None, type=str, help='Pre-trained ASR model to initialize encoder.'),
    "enc-init-mods": dict(default='encoder.,ctc.', type=split_string(','), help='List of encoder modules to initialize, separated by a comma.'),
    "dec-init": dict(default=None, type=str, help='Pre-trained ASR, MT or LM model to initialize decoder.'),
    "dec-init-mods": dict(default='decoder.', type=split_string(','), help='List of decoder modules to initialize, separated by a comma.'),

    # Encoder
    "use-frontend": dict(type=strtobool, default=False, help='The flag to switch to use frontend system.'),
    "num-encs": dict(default=1, type=int, help='Number of encoders in the model.'),
    "freeze-encoder": dict(type=int, default=0, help="Number of epoch for which the encoder is not updated"),

    # Decoder
    "context-residual": dict(default=False, type=strtobool, nargs='?', help='The flag to switch to use context vector residual in the decoder network'),

    # Loss
    "ctc_type": dict(default='warpctc', type=str, choices=['builtin', 'warpctc'], help='Type of CTC implementation to calculate loss.'),
    "mtlalpha": dict(default=0.5, type=float, help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss '),
    "lsm-weight": dict(default=0.0, type=float, help='Label smoothing weight'),
    "alpha-scheduler": dict(default=None, nargs=5, type=float, help="Scheduler for mtlalpha (xmin, xmax, rho, delay, max_step)"),

    # Weighted Prediction Error
    "use-wpe": dict(type=strtobool, default=False, help='Apply Weighted Prediction Error'),
    "wtype": dict(
        default='blstmp', type=str, choices=AVAILABLE_E2E_RNNS,
        help='Type of encoder network architecture of the mask estimator for WPE.'),
    "wlayers": dict(type=int, default=2, help=''),
    "wunits": dict(type=int, default=300, help=''),
    "wprojs": dict(type=int, default=300, help=''),
    "wdropout-rate": dict(type=float, default=0.0, help=''),
    "wpe-taps": dict(type=int, default=5, help=''),
    "wpe-delay": dict(type=int, default=3, help=''),
    "use-dnn-mask-for-wpe": dict(type=strtobool, default=False, help='Use DNN to estimate the power spectrogram. This option is experimental.'),

    # Beamformer
    "use-beamformer": dict(type=strtobool, default=True, help=''),
    "btype": dict(
        default='blstmp', type=str, choices=AVAILABLE_E2E_RNNS,
        help='Type of encoder network architecture of the mask estimator for Beamformer.'),
    "blayers": dict(type=int, default=2, help=''),
    "bunits": dict(type=int, default=300, help=''),
    "bprojs": dict(type=int, default=300, help=''),
    "badim": dict(type=int, default=320, help=''),
    "bnmask": dict(type=int, default=2, help='Number of beamforming masks, default is 2 for [speech, noise].'),
    "ref-channel": dict(type=int, default=-1, help='The reference channel used for beamformer. By default, the channel is estimated by DNN.'),
    "bdropout-rate": dict(type=float, default=0.0, help=''),

    # Feature transform: Normalization
    "stats-file": dict(type=str, default=None, help='The stats file for the feature normalization'),
    "apply-uttmvn": dict(type=strtobool, default=True, help='Apply utterance level mean variance normalization.'),
    "uttmvn-norm-means": dict(type=strtobool, default=True, help=''),
    "uttmvn-norm-vars": dict(type=strtobool, default=False, help=''),

    # Feature transform: Fbank
    "fbank-fs": dict(type=int, default=16000, help='The sample frequency used for the mel-fbank creation.'),
    "n-mels": dict(type=int, default=80, help='The number of mel-frequency bins.'),
    "fbank-fmin": dict(type=float, default=0., help=''),
    "fbank-fmax": dict(type=float, default=None, help=''),

}


def get_parser(parser=None, required=True):
    """Get default arguments."""

    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an automatic speech recognition (ASR) model on one CPU, one or multiple GPUs",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter
        )

    # Add additional config files
    for i in range(1, 4):
        parser.add(
            f"--config{'' if i == 1 else i}",
            is_config_file=True,
            help=f"Config file path {'' if i == 1 else '(overwrites settings in other config files)'}"
        )

    for arg, opts in CONFIG.items():
        parser.add_argument(f"--{arg}", **opts)

    return parser


def main(cmd_args):
    """Run the main training function."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    if args.model_class is None:
        model_class = "espnet.nets.pytorch_backend.e2e_asr:E2E"
    else:
        model_class = args.model_class
    model_class = dynamic_import(model_class)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    count_gpus(args)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    args.char_list = load_dictionary(args)

    # train

    if not args.v1:
        from espnet.mtl.train import train

    else:
        from espnet.asr.pytorch_backend.asr import train

    train(args)


if __name__ == '__main__':
    main(sys.argv[1:])
