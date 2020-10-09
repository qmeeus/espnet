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
from operator import itemgetter

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend_.rnn.encoders import encoder_for
from espnet.nets.scorers.ctc import CTCPrefixScorer


CTC_LOSS_THRESHOLD = 1e4


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, **metrics):
        """Report at every step.
        Expected: loss_ctc, loss_att, accuracy, cer_ctc, cer, wer, loss
        """
        for name, value in metrics.items():
            reporter.report({name: value}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    LOSS_NAMES = ["loss", "loss_ctc", "loss_att"]
    METRIC_NAMES = ["accuracy", "cer_ctc", "ter_ctc"]

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

        # Setup multitask learning loss
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"

        # Remove unused metrics
        if self.mtlalpha == 1:
            if "loss_att" in self.LOSS_NAMES:
                self.LOSS_NAMES.remove("loss_att")
            if "accuracy" in self.METRIC_NAMES:
                self.METRIC_NAMES.remove("accuracy")
        elif self.mtlalpha == 0:
            if "loss_ctc" in self.LOSS_NAMES:
                self.LOSS_NAMES.remove("loss_ctc")
            if "cer_ctc" in self.METRIC_NAMES:
                self.METRIC_NAMES.remove("cer_ctc")
                self.METRIC_NAMES.remove("ter_ctc")

        self.etype = args.etype
        self.verbose = args.verbose

        # NOTE: for self.build method
        self.char_list = args.char_list = getattr(args, "char_list", None)
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = self.eos = len(self.char_list) - 1

        # subsample info
        self.subsample = get_subsample(args, mode='asr', arch='rnn')

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        self.frontend = None

        # encoder
        self.encoder = encoder_for(args, idim, self.subsample)
        # ctc
        self.ctc = ctc_for(args, odim)
        # attention
        attention = att_for(args)
        # decoder
        self.decoder = decoder_for(args, odim, self.sos, self.eos, attention, labeldist)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -1e10

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        self.decoder.init_weights()

    def _forward(self, xs_pad, ilens, ys_pad):

        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        logging.debug(f"Input size: {hs_pad.size()} Output size: {ys_pad.size()}")

        # 1. Encoder
        hs_pad, hlens, _ = self.encoder(hs_pad, hlens)
        logging.debug(f"Enc out: {hs_pad.size()}")

        # 2. CTC
        ctc_out = self.ctc(hs_pad)

        # 3. Decoder
        prev_output_tokens = self.decoder.get_prev_output_tokens(ys_pad)
        dec_out, att, states = self.decoder(hs_pad, hlens, prev_output_tokens)

        return hs_pad, hlens, ctc_out, dec_out, att, states


    def forward(self, xs_pad, ilens, ys_pad, calc_metrics=False, compat_on=True):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """

        alpha = self.mtlalpha

        # HACK
        if compat_on:
            calc_metrics = True

        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
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

    def log_metrics(self, metrics):
        # FIXME: HACKY for compatibility with self.reporter
        # HACK: Convert to non tensors if necessary
        metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        loss = metrics["loss"]
        if loss < CTC_LOSS_THRESHOLD and not math.isnan(loss):
            self.reporter.report(**metrics)
        else:
            logging.debug('loss (=%f) is not correct', loss.item())


    def compute_metrics(self, hs_pad, hlens, ys_pad):
        # 1. compute cer without beam search
        cer_ctc = self.compute_cer_ctc(hs_pad, ys_pad)
        ter_ctc = self.compute_ter_ctc(hs_pad, ys_pad)

        # 2. compute cer/wer
        wer, cer = self.compute_wer_cer(hs_pad, hlens, ys_pad)

        return {"cer_ctc": cer_ctc, "ter_ctc": ter_ctc, "wer": wer, "cer": cer}


    def compute_loss(self, hs_pad, hlens, ys_pad):
        alpha = self.mtlalpha

        # 1. CTC loss
        loss_ctc = (
            self.ctc.compute_loss(hs_pad, hlens, ys_pad)
            if alpha > 0 else None
        )

        # 2. attention loss
        loss_att, accuracy, _ = (
            self.decoder.compute_loss(hs_pad, hlens, ys_pad)
            if alpha < 1 else [None] * 3
        )

        # 3. total loss
        loss = alpha * (loss_ctc or 0) + (1 - alpha) * (loss_att or 0)

        return {
            "loss": loss,
            "loss_ctc": loss_ctc.item() if loss_ctc else None,
            "loss_att": loss_att.item() if loss_att else None,
            "accuracy": accuracy
        }

    def compute_cer_ctc(self, hs_pad, ys_pad):
        if self.mtlalpha == 0 or self.char_list is None:
            return

        cers = []

        y_hats = self.ctc.argmax(hs_pad).data
        for i, y in enumerate(y_hats):
            y_hat = [x[0] for x in groupby(y)]
            y_true = ys_pad[i]

            seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
            seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
            seq_hat_text = seq_hat_text.replace(self.blank, '')
            seq_true_text = "".join(seq_true).replace(self.space, ' ')

            hyp_chars = seq_hat_text.replace(' ', '')
            ref_chars = seq_true_text.replace(' ', '')
            if len(ref_chars) > 0:
                cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            return sum(cers) / len(cers) if cers else None

    def compute_ter_ctc(self, hs_pad, ys_pad):
        if self.mtlalpha == 0 or self.char_list is None:
            return

        ters = []
        blank_index = self.char_list.index(self.blank)
        pad_index = -1
        ys_true = ys_pad.tolist()
        ys_hat = self.ctc.argmax(hs_pad).tolist()
        for i, (y_hat, y_true) in enumerate(zip(ys_hat, ys_true)):
            y_hat = list(filter(lambda x: x != blank_index, map(itemgetter(0), groupby(y_hat))))
            y_true = list(filter(lambda x: x != pad_index, y_true))

            if self.eos in y_hat:
                y_hat = y_hat[:y_hat.index(self.eos)]

            if len(y_true):
                ters.append(editdistance.eval(y_hat, y_true) / len(y_true))

            return sum(ters) / len(ters) if ters else None

    def compute_wer_cer(self, hs_pad, hlens, ys_pad):

        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(hs_pad).data
            else:
                lpz = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.decoder.recognize_beam_batch(
                hs_pad, torch.tensor(hlens), lpz,
                self.recog_args, self.char_list,
                self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)

        return wer, cer

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()

        # subsample frame
        x = x[::self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        ilens = torch.as_tensor([x.shape[0]], device=p.device)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(hs, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = hs, ilens

        # 1. encoder
        hs, _, _ = self.encoder(hs, hlens)
        return hs.squeeze(0)

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.decoder.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm)
        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = torch.as_tensor(np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64))

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(xs_pad, ilens)
            hs_pad, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.encoder(hs_pad, hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs_pad)
            normalize_score = False
        else:
            lpz = None
            normalize_score = True

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.decoder.recognize_beam_batch(hs_pad, hlens, lpz, recog_args, char_list,
                                          rnnlm, normalize_score=normalize_score)

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forward only in the frontend stage.

        :param ndarray xs: input acoustic feature (T, C, F)
        :return: enhaned feature
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise RuntimeError('Frontend does\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

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
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
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
