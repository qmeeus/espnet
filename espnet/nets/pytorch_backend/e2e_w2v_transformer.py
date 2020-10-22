# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import numpy as np
import editdistance
import logging
import math
import torch

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.losses import MaskedMSELoss
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_w2v import Decoder
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.utils.torch_utils import load_pretrained_embedding_from_file
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as BaseE2E


EMB_PATH = "/esat/spchdisk/scratch/qmeeus/repos/espnet/egs/cgn/asr1/data/lang_word/w2v_small.txt"
EMB_DIM = 100


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
        parser = BaseE2E.add_arguments(parser)
        group = parser._action_groups[-1]
        group.add_argument("--emb-path", type=str, default=EMB_PATH)
        group.add_argument("--emb-dim", type=int, default=EMB_DIM)
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
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
            input_layer=load_pretrained_embedding_from_file(
                args.emb_path, args.char_list, freeze=True
            )
        )

    def build_criterion(self, args):
        return MaskedMSELoss()

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

        from sklearn.neighbors import KNeighborsClassifier

        nneighbors = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(
            self.decoder.embed[0].weight.cpu(), np.arange(len(self.char_list))
        )

        loss_fn = MaskedMSELoss(reduction='none')

        self.eval()
        with torch.no_grad():

            output = {}

            # Encoder
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
            hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
            hlens = hs_mask.sum(-1)

            # Decoder
            ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id, ys_out_padding=self.pad)
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            pred_lens = pred_mask.sum(-1)

            # Attention
            attention_weights = {}
            for name, module in self.named_modules():
                if isinstance(module, MultiHeadedAttention):
                    attention_weights[name] = module.attn.cpu().numpy()

            # Loss
            ys_out_emb = self.decoder.embed[0](ys_out_pad)
            loss = loss_fn(pred_pad, ys_out_emb, (ys_pad != self.ignore_id).sum(-1) + 1)
            pred_pad = pred_pad.detach().cpu().numpy()
            output["loss"] = loss.detach().cpu().numpy()

            # Decoding with 1NN
            token2str = self.tokens_to_string
            target_sentences = list(map(token2str, [y[y != self.ignore_id] for y in ys_pad]))
            bs, olen, size = pred_pad.shape
            predicted_tokens = nneighbors.predict(pred_pad.reshape(bs * olen, size)).reshape(bs, olen)
            predicted_sentences = list(map(token2str, predicted_tokens))
            output["prediction_str"] = predicted_sentences
            output["accuracy"], output["wer"] = [], []
            for target_sentence, predicted_sentence in zip(target_sentences, predicted_sentences):
                logging.info(f"Target: {target_sentence}")
                logging.info(f"Prediction: {predicted_sentence}")
                words_true, words_pred = (sent.split() for sent in (target_sentence, predicted_sentence))

                output["accuracy"].append(
                    np.mean([word_true == word_pred
                    for word_true, word_pred in zip(words_true, words_pred)])
                )

                output["wer"].append(editdistance.eval(words_true, words_pred) / len(words_true))

            return pred_pad, attention_weights, output

    def log_metrics(self, metrics):
        # FIXME: HACKY for compatibility with self.reporter
        # HACK: Convert to non tensors if necessary
        metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        loss = metrics["loss"]
        if not math.isnan(loss):
            self.reporter.report(**metrics)
        else:
            import ipdb; ipdb.set_trace()
            logging.debug('loss (=%f) is not correct', loss.item())

    def tokens_to_string(self, tokens):
        tokens = list(tokens)
        if self.eos in tokens:
            tokens = tokens[:tokens.index(self.eos)]
        return " ".join(self.char_list[token] for token in tokens if token != self.pad)