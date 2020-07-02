import numpy as np
import editdistance
import logging
import math
import fairseq
import torch

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.losses import MaskedMSELoss
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_w2v import Decoder
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.utils.torch_utils import load_pretrained_embedding_from_file
from espnet.nets.pytorch_backend.e2e_w2v_transformer import E2E as BaseE2E
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as ASRE2E


class Teacher(torch.nn.Module):

    def __init__(self, model_string):
        super(Teacher, self).__init__()
        self.model = torch.hub.load(*self.parse_model_string(model_string))
        self.model.eval()
        self.freeze()
        self.to('cpu')

    def forward(self, tokens):
        return self.model.extract_features(tokens.to('cpu')).to(tokens.device)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def parse_model_string(model_string):
        parts = model_string.split("/")
        return "/".join(parts[:-1]), parts[-1]


class E2E(BaseE2E):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    LOSS_NAMES = ["loss"]
    METRIC_NAMES = []

    @staticmethod
    def add_arguments(parser):
        parser = ASRE2E.add_arguments(parser)
        group = parser._action_groups[-1]
        group.add_argument("--pretrained-model", type=str, default="pytorch/fairseq/xlmr.base")
        group.add_argument("--emb-dim", type=int, default=768)
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        args.mtlalpha = 0.
        args.report_cer = args.report_wer = False
        super(E2E, self).__init__(idim, odim, args, ignore_id=-1)

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
            input_layer=Teacher(args.pretrained_model)
        )

    def forward(self, xs_pad, ilens, ys_pad, calc_metrics=False, compat_on=True):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # HACK
        if compat_on:
            calc_metrics = True

        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        hlens = hs_mask.sum(-1)

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id, ys_out_padding=self.pad)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        self.pred_pad = pred_pad

        # 3. compute attention loss
        ys_out_emb = self.decoder.embed[0](ys_out_pad)
        loss_att = self.criterion(pred_pad, ys_out_emb, (ys_pad != self.ignore_id).sum(-1) + 1)
        losses = {"loss": loss_att}

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

    def predict(self, xs_pad, ilens, ys_pad, ylens):
        raise NotImplementedError

    def tokens_to_string(self, tokens):
        raise NotImplementedError