# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from itertools import groupby
import logging
import math

import numpy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as ASRTransformer
from espnet.nets.pytorch_backend.losses import MaskedMSELoss

class E2E(ASRTransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        parser = ASRTransformer.add_arguments(parser)
        group = parser.add_argument_group("teacher model setting")
        group.add_argument(
            "--teacher-model",
            default="wietsedv/bert-base-dutch-cased",
            type=str,
            help="Valid pretrained model from huggingface to serve as teacher model. "
            "See here for examples: https://huggingface.co/models",
        )
        group.add_argument(
            "--teacher-odim",
            default=30000,
            type=int,
            help="The output dimension of the teacher model (vocab size)."
        )
        group.add_argument(
            "--teacher-adim",
            default=768,
            type=int,
            help="The output dimension of the teacher model (vocab size)."
        )
        group.add_argument(
            "--mlm-probability",
            default=.7,
            type=float,
            help="The masking probability"
        )
        group.add_argument(
            "--alpha_mse",
            default=0.,
            type=float,
            help="The scaling factor for the MSE loss between the teacher and the student logits."
        )
        group.add_argument(
            "--alpha_ce",
            default=.5,
            type=float,
            help="The scaling factor for the KLD loss between the teacher and the student logits."
        )
        group.add_argument(
            "--alpha_mlm",
            default=.5,
            type=float,
            help="The scaling factor for the MLM loss."
        )
        group.add_argument(
            "--alpha_cos",
            default=0.,
            type=float,
            help="The scaling factor for the Cosine loss between the teacher and the student hidden states."
        )
        group.add_argument(
            "--temperature",
            default=2.,
            type=float,
            help="The scaling factor for the Cosine loss between the teacher and the student hidden states."
        )
        group.add_argument(
            "--restrict_ce_to_mask",
            default=False,
            type=bool,
            help="If true, compute the distilation loss only the [MLM] prediction distribution."
        )
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        args.report_cer = args.report_wer = False
        super(E2E, self).__init__(idim, odim, args, ignore_id=-1)
        self.teacher_model, self.teacher_tokenizer = self.load_teacher(args)
        self.collator = DataCollatorForLanguageModeling(
            self.teacher_tokenizer, mlm_probability=args.mlm_probability)
        self.mask_token = self.teacher_tokenizer.mask_token_id
        self.vocabulary = args.char_list
        self.alpha_mse = args.alpha_mse
        self.alpha_ce = args.alpha_ce
        self.alpha_cos = args.alpha_cos
        self.restrict_ce_to_mask = args.restrict_ce_to_mask
        
        self.temperature = args.temperature
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0:
            self.mse_loss_fct = MaskedMSELoss(reduction="sum")
        if self.alpha_cos > 0:
            self.cos_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        self.teacher_odim = args.teacher_odim

    def convert_tokens_to_string(self, tokens):
        if type(tokens[0]) == int:
            return "".join([
                self.vocabulary[i] for i in tokens if i != self.ignore_id
            ]).replace("▁", " ").strip()
        else:
            return [self.convert_tokens_to_string(ids) for ids in tokens.tolist()]

    def load_teacher(self, args):
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        teacher_model = AutoModelForMaskedLM.from_pretrained(args.teacher_model)
        assert args.teacher_odim == teacher_model.config.vocab_size
        assert args.teacher_adim == teacher_model.config.hidden_size
        
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.to(next(self.decoder.parameters()).device)
        return teacher_model, teacher_tokenizer

    def build_decoder(self, odim, args):
        odim = args.teacher_odim
        return super(E2E, self).build_decoder(odim, args)

    def build_criterion(self, odim, ignore_id, args):
        return super(E2E, self).build_criterion(args.teacher_odim, -100, args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad
        device = hs_pad.device

        # 2. forward decoder
        texts = self.convert_tokens_to_string(ys_pad)
        tokens = self.teacher_tokenizer(texts)["input_ids"]
        decoder_inputs = self.collator(list(map(torch.tensor, tokens)))

        ys_in_pad, ys_out_pad = decoder_inputs["input_ids"].to(device), decoder_inputs["labels"].to(device)
        ys_mask = (ys_in_pad != self.teacher_tokenizer.pad_token_id)

        with torch.no_grad():
            t_logits, t_hidden_states = self.teacher_model(
                input_ids=ys_in_pad, 
                attention_mask=ys_mask, 
                output_hidden_states=True
            )

        s_logits, pred_mask, s_hidden_states = self.decoder(
            ys_in_pad, ys_mask.unsqueeze(1), hs_pad, hs_mask, return_hidden=True
        )
        
        if self.restrict_ce_to_mask:
            mask = (ys_out_pad > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        else:
            mask = ys_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        # 3. compute attention loss
        loss_label = self.mlm_loss_fct(s_logits.view(-1, self.teacher_odim), ys_out_pad.view(-1))
        loss_att = loss_label

        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            ) * self.temperature ** 2
        )

        loss_att += loss_ce * self.alpha_ce

        if self.alpha_mse > 0:
            loss_mse = self.mse_loss(decoder_proj, teacher_out, ys_mask.sum(-1))
            loss_att += loss_mse * self.alpha_mse

        if self.alpha_cos > 0:
            t_hidden_states = t_hidden_states[-1]
            mask = ys_mask.unsqueeze(-1).expand_as(s_hidden_states)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss_att += self.alpha_cos * loss_cos

        self.acc = th_accuracy(
            s_logits.view(-1, self.teacher_odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = s_logits.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss
