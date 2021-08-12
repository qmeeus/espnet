# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""
import chainer
import logging
import math
import torch

from chainer import reporter
from itertools import groupby
from operator import itemgetter
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    default_data_collator
)

from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as ASRTransformer
from espnet.nets.pytorch_backend.losses import MaskedMSELoss


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, **kwargs):
        for key, value in kwargs.items():
            reporter.report({key: value}, self)


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
            "--wwm",
            default=False,
            type=bool,
            help="Mask whole words"
        )
        group.add_argument(
            "--no-mask-teacher-inputs",
            default=False,
            type=bool,
            help="Pass unmasked token to teacher rather than masked inputs."
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
        Collator = DataCollatorForWholeWordMask if args.wwm else DataCollatorForLanguageModeling
        self.collator = Collator(
            self.teacher_tokenizer, mlm_probability=args.mlm_probability)
        self.mask_token = self.teacher_tokenizer.mask_token_id
        self.vocabulary = args.char_list
        self.alpha_mlm = args.alpha_mlm
        self.alpha_mse = args.alpha_mse
        self.alpha_ce = args.alpha_ce
        self.alpha_cos = args.alpha_cos
        self.restrict_ce_to_mask = args.restrict_ce_to_mask
        self.mask_teacher_inputs = not(args.no_mask_teacher_inputs)
        
        self.temperature = args.temperature
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="sum")

        self.teacher_odim = args.teacher_odim

        self.reporter = Reporter()

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

        bs = xs_pad.size(0)

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

        teacher_pad_token_id = self.teacher_tokenizer.pad_token_id
        ys_in_pad, ys_out_pad = decoder_inputs["input_ids"].to(device), decoder_inputs["labels"].to(device)
        ys_mask = (ys_in_pad != teacher_pad_token_id)

        with torch.no_grad():
            teacher_inputs = (
                ys_in_pad if self.mask_teacher_inputs 
                else torch.nn.utils.rnn.pad_sequence(
                    list(map(torch.tensor, tokens)), 
                    padding_value=teacher_pad_token_id,
                    batch_first=True
                ).to(device)
            )
            teacher_mask = (teacher_inputs != teacher_pad_token_id)
            t_logits, t_hidden_states = self.teacher_model(
                input_ids=teacher_inputs, 
                attention_mask=teacher_mask, 
                output_hidden_states=True,
                return_dict=False
            )

            t_hidden_states = t_hidden_states[-1]
        
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
        loss_att = loss_label * self.alpha_mlm

        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            ) * self.temperature ** 2
        )

        loss_att += loss_ce * self.alpha_ce
        loss_mse, loss_cos = None, None

        if self.alpha_mse > 0 or self.alpha_cos > 0:
            assert s_hidden_states.size() == t_hidden_states.size()
            mask = ys_mask.unsqueeze(-1).expand_as(s_hidden_states)
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

        if self.alpha_mse > 0:
            assert s_hidden_states.size() == t_hidden_states.size()
            loss_mse = self.mse_loss_fct(s_hidden_states_slct, t_hidden_states_slct) / s_hidden_states_slct.size(0)
            loss_att += loss_mse * self.alpha_mse

        if self.alpha_cos > 0:

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target) / bs
            loss_att += self.alpha_cos * loss_cos

        self.acc = th_accuracy(
            s_logits.view(-1, self.teacher_odim), ys_out_pad, ignore_label=-100
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
                loss_ctc=loss_ctc_data,
                loss_att=loss_att_data,
                loss_mlm=float(loss_label),
                loss_ce=float(loss_ce),
                loss_mse=float(loss_mse) if loss_mse else None,
                loss_cos=float(loss_cos) if loss_cos else None,
                acc=self.acc, 
                cer_ctc=cer_ctc,
                cer=cer, wer=wer, 
                loss=loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss


    def encode(self, xs_pad, ilens, p_thres=.999, blank_multiplier=1):
        _, _, s_hidden_states, output_lengths = self.predict(
            xs_pad, ilens, p_thres=.999, blank_multiplier=blank_multiplier)
        return s_hidden_states, output_lengths

    def predict(self, xs_pad, ilens, K=10, p_thres=.99, blank_multiplier=1):
        """E2E encode.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :return: encoded features
        :rtype: torch.Tensor
        :return: output_lengths
        :rtype: torch.Tensor
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        device = hs_pad.device

        # 2. forward ctc
        texts, ctc_preds = self._ctc_predict(hs_pad, p_thres, blank_multiplier)
            
        # 2. forward decoder
        decoder_inputs = self.teacher_tokenizer(texts, return_tensors="pt", padding=True)

        ys_in_pad = decoder_inputs["input_ids"].to(device)
        attention_mask = decoder_inputs["attention_mask"].to(device)
        if K > 0:
            logits, pred_mask, hidden_states = self._dec_predict(ys_in_pad, attention_mask, hs_pad, hs_mask, p_thres=p_thres, K=K)
        else:
            logits, pred_mask, hidden_states = self.decoder(
                ys_in_pad, attention_mask.unsqueeze(1), hs_pad, hs_mask, return_hidden=True
            )

        hidden_states.masked_fill_(~attention_mask.unsqueeze(-1).bool(), 0.)
        return F.log_softmax(logits, dim=-1).detach(), ctc_preds, hidden_states.detach(), attention_mask.sum(-1).detach()

    def _ctc_predict(self, hs_pad, p_thres, blank_multiplier):

        batch_size = hs_pad.size(0)
        ctc_probs = self.ctc.softmax(hs_pad, blank_multiplier)
        ctc_probs, ctc_ids = ctc_probs.max(dim=-1)
        
        texts = []
        ctc_preds = []
        mask_token_id = self.vocabulary.index("<unk>")
        for i in range(batch_size):
            y_hat, probs_hat = map(torch.tensor, zip(
                *map(lambda group: (group[0], max(map(itemgetter(1), group[1]))), groupby(
                    zip(ctc_ids[i], ctc_probs[i]), key=lambda x: x[0]
                ))
            ))

            y_idx = torch.nonzero(y_hat != 0).squeeze(-1)
            mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
            confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)            
            seq = y_hat[y_idx].clone()
            ctc_preds.append(y_hat[y_idx].clone().detach())
            seq[mask_idx] = mask_token_id
            texts.append("".join([
                self.vocabulary[token_id] 
                if token_id != mask_token_id else self.teacher_tokenizer.mask_token
                for token_id in seq
            ]).replace("▁", " "))

        return texts, ctc_preds

    def _dec_predict(self, ys_in_pad, attn_mask, hs_pad, hs_mask, K=10, p_thres=.99):
        
        mask_token_id = self.teacher_tokenizer.mask_token_id
        logits, pred_mask, hidden_states = self.decoder(
            ys_in_pad, attn_mask.unsqueeze(1), hs_pad, hs_mask, return_hidden=True
        )
        
        ps = torch.softmax(logits, -1)

        ys = torch.zeros_like(ys_in_pad)
        for i in range(hs_pad.size(0)):
            ps_, ys_ = ps[i].max(-1)
            mask = (attn_mask[i] & (ps_ < p_thres)).bool()
            K_ = min(K, mask.sum())
            ys_[mask] = mask_token_id
            indices = torch.arange(ys_.size(0))
            for k in range(K_):
                logits_, _ = self.decoder(
                    ys_.unsqueeze(0), attn_mask[i:i+1], hs_pad[i:i+1], hs_mask[i:i+1]
                )
                new_ps_, new_ys_ = torch.softmax(logits_, -1).max(-1)
                mask_ind = new_ps_[0, mask].argmax()
                ys_[indices[mask][mask_ind]] = new_ys_[0, mask][mask_ind]
                mask[indices[mask][mask_ind]] = False
            ys[i] = ys_
                
        # Remaining masked tokens
        logits, pred_mask, hidden_states = self.decoder(
            ys, attn_mask.unsqueeze(1), hs_pad, hs_mask, return_hidden=True
        )
        return logits, pred_mask, hidden_states
