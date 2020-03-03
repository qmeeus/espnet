import os
import logging
import subprocess
from collections.abc import Sequence
from distutils.util import strtobool as dist_strtobool
import sys
import torch
from distutils.version import LooseVersion

import numpy


def strtobool(x):
    # distutils.util.strtobool returns integer, but it's confusing,
    return bool(dist_strtobool(x))


def get_commandline_args():
    extra_chars = [' ', ';', '&', '(', ')', '|', '^', '<', '>', '?', '*',
                   '[', ']', '$', '`', '"', '\\', '!', '{', '}']

    # Escape the extra characters for shell
    argv = [arg.replace('\'', '\'\\\'\'')
            if all(char not in arg for char in extra_chars)
            else '\'' + arg.replace('\'', '\'\\\'\'') + '\''
            for arg in sys.argv]

    return sys.executable + ' ' + ' '.join(argv)


def is_scipy_wav_style(value):
    # If Tuple[int, numpy.ndarray] or not
    return (isinstance(value, Sequence) and len(value) == 2 and
            isinstance(value[0], int) and
            isinstance(value[1], numpy.ndarray))


def assert_scipy_wav_style(value):
    assert is_scipy_wav_style(value), \
        'Must be Tuple[int, numpy.ndarray], but got {}'.format(
            type(value) if not isinstance(value, Sequence)
            else '{}[{}]'.format(type(value),
                                 ', '.join(str(type(v)) for v in value)))


def count_gpus(args):
    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(','))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(['nvidia-smi', '-L'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split('\n')) - 1
    else:
        is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion('1.2')
        if is_torch_1_2_plus and args.ngpu != 1:
            logging.debug("There are some bugs with multi-GPU processing in PyTorch 1.2+" +
                          " (see https://github.com/pytorch/pytorch/issues/21108)")
        ngpu = args.ngpu
        
    logging.info(f"ngpu: {ngpu}")
    args.ngpu = ngpu
