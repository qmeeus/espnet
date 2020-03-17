#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech recognition model (pytorch)."""

import argparse
import logging
import math
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from espnet.nets.pytorch_backend_.ctc import ctc_for
from espnet.nets.pytorch_backend_.rnn.attentions import att_for
from espnet.nets.pytorch_backend_.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend_.rnn.encoders_ import encoder_for

from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.nets_utils import get_subsample


class E2E(nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    LOSS_NAMES = ["loss", "loss_ctc", "loss_att"]
    METRIC_NAMES = ["cer_ctc", "cer", "wer"]

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
        group.add_argument('--etype', default='blstmp', type=str, choices=[
            'lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm', 
            'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'
        ], help='Type of encoder network architecture')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', default="1", type=str,  # FIXME: unclear and inelegant
                           help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                                'every y frame at 2nd layer etc.')
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2E attention setting")
        # attention
        group.add_argument('--atype', default='dot', type=str, choices=[
            'noatt', 'dot', 'add', 'location', 'coverage', 'coverage_location', 'location2d', 
            'location_recurrent', 'multi_head_dot', 'multi_head_add', 'multi_head_loc', 'multi_head_multi_res_loc'
        ], help='Type of attention architecture')
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
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2E encoder setting")
        group.add_argument('--dtype', default='lstm', type=str, choices=['lstm', 'gru'],
                           help='Type of decoder network architecture')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--sampling-probability', default=0.0, type=float,
                           help='Ratio of predicted labels fed back to decoder')
        group.add_argument('--lsm-type', const='', default='', type=str, nargs='?',
                           choices=['', 'unigram'],
                           help='Apply label smoothing with a specified distribution type')
        return parser

    def __init__(self, input_dim, output_dim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = self.eos = output_dim - 1

        # subsample info # HACK
        self.subsample = args.subsample = get_subsample(args, mode='asr', arch='rnn')

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(output_dim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # encoder
        self.encoder = encoder_for(input_dim, args)
        
        # ctc
        if self.mtlalpha > 0:
            self.ctc = ctc_for(args, output_dim)
        
        # attention & decoder
        if self.mtlalpha < 1:
            self.attention = att_for(args)        
            self.decoder = decoder_for(args, output_dim, self.sos, self.eos, self.attention, labeldist)

        # weight initialization
        self.init_weights()

        # # options for beam search
        # if args.report_cer or args.report_wer:
        #     self.recog_args = argparse.Namespace()
        #     for arg in ("beam_size", "penalty", "ctc_weight", "maxlenratio", "minlenratio", 
        #                 "lm_weight", 'rnnlm', 'nbest', 'sym_space', 'sym_blank'):
        #         setattr(self.recog_args, getattr(args, arg))
            
        #     self.report_cer = args.report_cer
        #     self.report_wer = args.report_wer

        # else:
        #     self.report_cer = False
        #     self.report_wer = False

    def init_weights(self):
        """Initialize weight like chainer.
        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        if hasattr(self, 'decoder'):
            self.decoder.init_weights()

    def forward(self, xs_pad, ilens, ys_pad, calc_metrics=False, compat_on=True):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """

        alpha = self.mtlalpha

        logging.debug(f"Input shape: {xs_pad.size()} output shape: {ys_pad.size()}")

        # 1. Encoder
        enc_out, enc_lens, _ = self.encoder(xs_pad, ilens)
        logging.debug(f"Enc_out shape: {enc_out.size()}")

        # 2. CTC predictions
        ctc_pred = (
            self.ctc(enc_out) 
            if self.ctc is not None else None
        )

        logging.debug(f"y_ctc shape: {ctc_pred.size()}")

        # 3. Encoder predictions
        dec_pred = (
            self.decoder(enc_out, enc_lens, ys_pad)
            if hasattr(self, 'decoder') else None
        )

        logging.debug(f"y_d shape: {dec_pred.size() if dec_pred is not None else 'na'}")

        return enc_out, enc_lens, ctc_pred, dec_pred

    def compute_loss(self, outputs, target, target_lengths):
        enc_out, enc_lens, ctc_pred, dec_pred = outputs        
        alpha = self.mtlalpha

        # 1. CTC loss
        loss_ctc = (
            self.ctc.compute_loss(ctc_pred, enc_lens, target, target_lengths) 
            if alpha > 0 else torch.tensor(0).type_as(enc_out)
        )
        
        # 2. attention loss
        loss_att = (
            self.decoder.compute_loss(dec_pred, target) 
            if alpha < 1 else torch.tensor(0).type_as(enc_out)
        )

        # 3. total loss
        loss = alpha * loss_ctc + (1 - alpha) * loss_att

        return loss, loss_ctc, loss_att

    def compute_metrics(self, outputs, target, target_lengths):
        # TODO: Simplify and improve readability
        enc_out, enc_lens, ctc_pred, dec_pred = outputs        

        cer_ctc, cer, wer = (
            self.ctc.compute_metrics(enc_out, enc_lens, target, self.char_list)
            if self.mtlalpha == 0 or self.char_list is None 
            else [torch.tensor(float('nan')).type_as(outputs)] * 3
        )

        return cer_ctc, cer, wer

    # def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
    #     """E2E attention calculation.

    #     :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
    #     :param torch.Tensor ilens: batch of lengths of input sequences (B)
    #     :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
    #     :return: attention weights with the following shape,
    #         1) multi-head case => attention weights (B, H, Lmax, Tmax),
    #         2) other case => attention weights (B, Lmax, Tmax).
    #     :rtype: float ndarray
    #     # TODO: Move and simplify
    #     """
    #     with torch.no_grad():
    #         # 0. Frontend
    #         if self.frontend is not None:
    #             hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
    #             hs_pad, hlens = self.feature_transform(hs_pad, hlens)
    #         else:
    #             hs_pad, hlens = xs_pad, ilens

    #         # 1. Encoder
    #         hpad, hlens, _ = self.encoder(hs_pad, hlens)

    #         # 2. Decoder
    #         att_ws = self.decoder.calculate_all_attentions(hpad, hlens, ys_pad)

    #     return att_ws
