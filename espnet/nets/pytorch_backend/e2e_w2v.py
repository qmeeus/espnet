#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech recognition model (pytorch)."""

from __future__ import division

import argparse
import logging
import math
import os

import editdistance

import chainer
import numpy as np
import six
import torch

from itertools import groupby
from chainer import reporter

from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for

from espnet.nets.pytorch_backend.rnn.decoders_ import decoder_for


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, **metrics):
        """Report at every step.
        Expected: loss_ctc, loss_att, accuracy, cer_ctc, cer, wer, loss
        """
        for name, value in metrics.items():
            reporter.report({name: value}, self)


class E2E(torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2E encoder setting")
        # encoder
        group.add_argument('--etype', default='blstmp', type=str,
                           choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                           help='Type of encoder network architecture')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', default="1", type=str,
                           help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                                'every y frame at 2nd layer etc.')
        group.add_argument('--encoder-dropout', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2E attention setting")
        # attention
        group.add_argument('--atype', default='dot', type=str,
                           choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                    'coverage_location', 'location2d', 'location_recurrent',
                                    'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                    'multi_head_multi_res_loc'],
                           help='Type of attention architecture')
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--awin', default=5, type=int,
                           help='Window size for location2d attention')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--aconv-chans', default=-1, type=int,
                           help='Number of attention convolution channels \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--aconv-filts', default=100, type=int,
                           help='Number of attention convolution filters \
                           (negative value indicates no location-aware attention)')
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2E encoder setting")
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru'],
                           help='Type of decoder network architecture')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--decoder-dropout', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--sampling-probability', default=0.0, type=float,
                           help='Ratio of predicted labels fed back to decoder')
        group.add_argument('--lsm-type', const='', default='', type=str, nargs='?',
                           choices=['', 'unigram'],
                           help='Apply label smoothing with a specified distribution type')
        return parser

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        args.char_list = getattr(args, "char_list", None)
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.reporter = Reporter()

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = self.eos = self.char_list.index("</s>")  #len(self.char_list) - 1
        self.pad = self.char_list.index("<pad>")

        # subsample info
        self.subsample = get_subsample(args, mode='asr', arch='rnn')

        # encoder
        self.encoder = encoder_for(args, idim, self.subsample)
        # attention
        self.attention = att_for(args)
        # decoder
        self.decoder = decoder_for(args, odim, self.sos, self.eos, self.attention)

        # weight initialization
        self.init_like_chainer()

        self.report_cer = False
        self.report_wer = False

        self.logzero = -1e10

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self.encoder)
        lecun_normal_init_parameters(self.attention)
        self.decoder.init_weights()

    def forward(self, xs_pad, ilens, ys_pad, ylens=None, calc_metrics=False, compat_on=True):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """

        # HACK
        if compat_on:
            calc_metrics = True

        hs_pad, hlens = xs_pad, ilens

        logging.debug(f"Input size: {hs_pad.size()} Output size: {ys_pad.size()}")

        # 1. Encoder
        hs_pad, hlens, _ = self.encoder(hs_pad, hlens)

        logging.debug(f"Enc out: {hs_pad.size()}")

        # 2. Loss
        losses = self.compute_loss(hs_pad, hlens, ys_pad)

        # 3. Metrics HACK
        if calc_metrics:
            metrics = self.compute_metrics(hs_pad, hlens, ys_pad)
            output = dict(**losses, **metrics)

            if compat_on:
                # 4. Log metrics
                self.log_metrics(output)
            
            if not compat_on:
                return output

        if compat_on:
            return output["loss"]        

        return losses

    def evaluate(self, xs_pad, ilens, ys_pad):
        self.eval()
        with torch.no_grad():
            hs_pad, hlens, _ = self.encoder(xs_pad, ilens)
            y_pred, att_ws = self.decoder(hs_pad, hlens, ys_pad)
            return y_pred, att_ws

    def log_metrics(self, metrics):
        # FIXME: HACKY for compatibility with self.reporter
        # HACK: Convert to non tensors if necessary
        metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        loss = metrics["loss"]
        if not math.isnan(loss):
            self.reporter.report(**metrics)
        else:
            logging.debug('loss (=%f) is not correct', loss.item())


    def compute_metrics(self, *args):
        return {}


    def compute_loss(self, hs_pad, hlens, ys_pad):
        
        # 2. attention loss
        loss, accuracy, _ = self.decoder.compute_loss(hs_pad, hlens, ys_pad)

        return {"loss": loss, "accuracy": accuracy}

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        from espnet.asr.asr_utils import PlotAttentionReport
        return PlotAttentionReport

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            # 0. Frontend
            hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            hpad, hlens, _ = self.encoder(hs_pad, hlens)

            # 2. Decoder
            att_ws = self.decoder.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen
