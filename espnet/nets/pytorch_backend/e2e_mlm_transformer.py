import numpy as np
import editdistance
import logging
import math
import fairseq
import torch
from torch import nn
from torch.nn import functional as F
from random import random
from itertools import groupby
from operator import itemgetter

from transformers import DataCollatorForLanguageModeling
from transformers import AutoModel, AutoTokenizer

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos, mask_uniform
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_w2v import Decoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.utils.torch_utils import load_pretrained_embedding_from_file
from espnet.nets.pytorch_backend.e2e_w2v_transformer import E2E as BaseE2E
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as ASRE2E
from espnet.nets.e2e_asr_common import tokens2text
from espnet.nets.pytorch_backend.nets_utils import pad_list


class MaskedMSELoss(nn.Module):
    
    def __init__(self, reduction='sum'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, target_mask):
        mse = F.mse_loss(pred, target, reduction='none').mean(-1)
        mse.masked_fill_(target_mask, 0)
        sequence_loss = mse.sum(-1)
        if self.reduction == 'sum':
            return sequence_loss.sum()
        elif self.reduction == 'mean':
            return (sequence_loss / target_lengths).mean()
        else:
            return sequence_loss


def target_mask(ys_pad, ignore_id=0, mask_eos=True):
    mask = (ys_pad == ignore_id).all(axis=-1)
    maxlen = mask.size(1)
    if mask_eos:
        for item_idx, token_idx in enumerate((~mask).sum(-1)):
            mask[item_idx, token_idx - 1] = False
    m = subsequent_mask(mask.size(-1), device=mask.device).unsqueeze(0)
    return mask.unsqueeze(-2) & m

class E2E(BaseE2E):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    LOSS_NAMES = ["loss", "loss_att", "loss_ctc"]
    METRIC_NAMES = []

    @staticmethod
    def add_arguments(parser):
        parser = ASRE2E.add_arguments(parser)
        group = parser._action_groups[-1]
        group.add_argument("--teacher-model", type=str, default="distilbert-base-multilingual-cased")
        group.add_argument("--emb-dim", type=int, default=768)
        group.add_argument("--mlm-probability", type=float, default=.8)
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        args.report_cer = args.report_wer = False
        super(E2E, self).__init__(idim, odim, args, ignore_id=-1)
        self.teacher_model, self.teacher_tokenizer = self.load_teacher(args.teacher_model)
        self.mask_token = self.teacher_tokenizer.mask_token_id
        self.ignore_id = -100  # hardcoded in transformers
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.teacher_tokenizer, mlm=True, mlm_probability=args.mlm_probability
        )
        assert self.decoder_mode == "maskctc", "`self.decoder_mode` can only be maskctc for this class"

    @staticmethod
    def load_teacher(teacher_model_identifier):
        tokenizer = AutoTokenizer.from_pretrained(teacher_model_identifier)
        teacher = AutoModel.from_pretrained(teacher_model_identifier)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher, tokenizer

    def get_sos_embedding(self):
        return self.teacher_model(
            self.teacher_tokenizer.encode(
                "",
                add_special_tokens=True,
                truncation="longest_first",
                return_tensors='pt'
            )[:, :1].to(self.teacher_model.device)
        )[0].squeeze()

    def teacher_encode(self, sentences, add_special_tokens=True):
        # NOTE: Probably super slow

        tensors = [
            self.teacher_model(
                self.teacher_tokenizer.encode(
                    sentence,
                    add_special_tokens=add_special_tokens,
                    truncation="longest_first",
                    return_tensors='pt'
                ).to(self.teacher_model.device)
            )[0].squeeze(0) for sentence in sentences
        ]

        return pad_list(tensors, 0)

    def build_decoder(self, odim, args):
        return Decoder(
            odim=args.emb_dim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            input_layer="linear"
        )

    def build_criterion(self, args):
        return MaskedMSELoss()

    def forward(self, xs_pad, ilens, ys_pad, calc_metrics=False, compat_on=True, ctc_forcing=True):
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
        # HACK
        if compat_on:
            calc_metrics = True

        losses = dict()

        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        hlens = hs_mask.sum(-1)
        batch_size = len(hlens)

        tokens = ys_pad

        # NOTE: not the most efficient way to achieve this.
        #   Should consider passing texts directly to forward (espnet2)
        texts = [
            tokens2text(
                tokens=y.tolist(),
                char_list=self.char_list,
                ignore_indices=(0, -1),
                return_sentence=True,
                is_ctc=False
            ) for y in tokens
        ]

        # 2a. ctc
        if hasattr(self, 'ctc'):
            assert self.char_list is not None
            enc_out = hs_pad.view(batch_size, -1, self.adim)
            loss_ctc = self.ctc.compute_loss(enc_out, hlens, ys_pad)
            ctc_out = self.ctc.argmax(enc_out).data
            losses["loss_ctc"] = loss_ctc
            if ctc_forcing:
                mask_forcing = torch.rand(batch_size) < self.sampling_probability
                logging.debug(f"CTC Forcing: Replacing {mask_forcing.sum()} items with CTC prediction")
                for i in torch.arange(batch_size)[mask_forcing]:
                    ctc_pred = tokens2text(
                        tokens=ctc_out[i].tolist(),
                        char_list=self.char_list,
                        ignore_indices=(0, -1),
                        return_sentence=True,
                        is_ctc=True
                    )
                    logging.debug(f"CTC Forcing: {texts[i]}  --> {ctc_pred}")
                    texts[i] = ctc_pred


        # 2b. forward decoder
        batch = self.data_collator(
            list(map(self.teacher_tokenizer.encode, texts))
        )

        ys_in_pad, ys_out_pad = batch["input_ids"].to(hs_pad.device), batch["labels"].to(hs_pad.device)
        ys_mask = (ys_in_pad != self.teacher_tokenizer.pad_token_id)
        ys_out_mask = (ys_out_pad != self.ignore_id)
        ys_out_pad.masked_fill_(ys_out_pad == self.ignore_id, self.teacher_tokenizer.pad_token_id)
        ys_in_pad = self.teacher_model(ys_in_pad, attention_mask=ys_mask)[0].to(ys_pad.device)
        ys_out_pad = self.teacher_model(ys_out_pad, attention_mask=ys_mask)[0].to(ys_pad.device)
        ys_mask = ys_mask.unsqueeze(1)

        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad, ys_out_mask)
        losses["loss_att"] = loss_att

        if hasattr(self, 'ctc'):
            losses["loss"] = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att
        else:
            losses["loss"] = loss_att

        # 3. Metrics HACK
        if calc_metrics:
            metrics = {}
            output = dict(**losses, **metrics)

            if compat_on:
                # 4. Log metrics
                self.log_metrics(output)

            if not compat_on:
                return output

        if compat_on:
            return output["loss"]

        return losses

    def scorers(self):
        raise NotImplementedError

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        raise NotImplementedError

    def predict(self, xs_pad, ilens):

        force_ctc = True

        assert not(force_ctc) or (hasattr(self, 'ctc') and self.char_list is not None)

        self.eval()
        with torch.no_grad():

            # 1. Forward encoder
            xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
            hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
            hlens = hs_mask.sum(-1)
            batch_size = len(hlens)

            if force_ctc:
                # 2. Forward CTC
                enc_out = hs_pad.view(batch_size, -1, self.adim)
                ctc_out = self.ctc.argmax(enc_out).data

                # 3. Transform CTC output in texts
                texts = [
                    tokens2text(
                        tokens=y.tolist(),
                        char_list=self.char_list,
                        ignore_indices=(0, -1),
                        return_sentence=True,
                        is_ctc=True
                    ) for y in ctc_out
                ]

                # 4. Encode the texts with the teacher model
                ys_pad = self.teacher_encode(texts).to(hs_pad.device)

                # 5. Forward decoder
                ys_in_pad, ys_out_pad = ys_pad[:, :-1].clone(), ys_pad[:, 1:].clone()
                ys_mask = target_mask(ys_in_pad)
                pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

                return pred_pad, pred_mask

            else:
                sos_embedding = self.get_sos_embedding()
                maxlen = enc_out.shape[1]
                ys = sos_embedding.repeat(batch_size, 1).view(batch_size, 1, -1)
                c = None

                # TODO: how to detect end vector?
                for i in range(maxlen):

                    # get nbest local scores and their ids
                    ys_mask = subsequent_mask(i + 1).repeat(batch_size, 1, 1).to(enc_out.device)
                    y, c = self.decoder.forward_one_step(ys, ys_mask, enc_out, cache=c)
                    ys = torch.cat([ys, y.unsqueeze(1)], dim=1)

                return ys, torch.tensor([False]).repeat(*ys.size())


    def recognize_maskctc(self, x, char_list=None, p_thres=.99):
        """Non-autoregressive decoding using Mask CTC.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :return: decoding result
        :rtype: list
        """

        self.eval()
        h = self.encode(x).unsqueeze(0)
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)

        y_hat, probs_hat = map(torch.tensor, zip(
            *map(lambda group: (group[0], max(map(itemgetter(1), group[1]))), groupby(
                zip(ctc_ids.squeeze(0), ctc_probs.squeeze(0)), key=lambda x: x[0]
            ))
        ))

        y_idx = torch.nonzero(y_hat != 0).squeeze(-1)

        char_mask = "_"
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        mask_num = len(mask_idx)

        tokens = y_hat[y_idx]
        tokens[mask_idx] = self.char_list.index("<mask>")
        sentence = "".join(map(self.char_list.__getitem__, tokens)).replace("▁", " ").replace("<mask>", "[MASK]")
        tokens_in = self.teacher_tokenizer.encode(sentence, return_tensors="pt").to(h.device)
        y_in_mask = (tokens_in != self.teacher_tokenizer.pad_token_id)
        y_in = self.teacher_model(tokens_in, attention_mask=y_in_mask)[0].to(h.device)

        if len(y_idx):
            logging.info(f"Masked inputs: {len(mask_idx)} ({len(mask_idx) / len(y_idx):.3%})")
        else:
            logging.info("Empty prediction")
        logging.info(f"ctc: {sentence}")

        pred, pred_mask = self.decoder(
            y_in, y_in_mask.unsqueeze(0), h, None
        )

        return pred, pred_mask




    def tokens_to_string(self, tokens):
        raise NotImplementedError
