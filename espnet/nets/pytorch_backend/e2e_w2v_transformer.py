# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import numpy as np
from argparse import Namespace
from distutils.util import strtobool
import editdistance
import logging
import math
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample, make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.losses import MaskedMSELoss
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_ import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.torch_utils import load_pretrained_embedding_from_file


EMB_PATH = "/esat/spchdisk/scratch/qmeeus/repos/espnet/egs/cgn/asr1/data/lang_word/w2v_small.txt"
EMB_DIM = 100


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    LOSS_NAMES = ["loss"]
    METRIC_NAMES = []

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')
        group.add_argument('--dropout-rate', default=0.0, type=float, help='Dropout rate')

        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
            
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )

        dec_input = load_pretrained_embedding_from_file(EMB_PATH, args.char_list, freeze=True)

        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            input_layer=dec_input,
            emb_dim=EMB_DIM
        )

        self.char_list = args.char_list
        self.sos = self.eos = self.char_list.index("</s>")
        self.pad = self.char_list.index("<pad>")

        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode='asr', arch='transformer')
        self.reporter = Reporter()

        self.criterion = MaskedMSELoss()

        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        self.ctc = None

        self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        # TODO: apparently, this does not change the weights of the embeddings
        initialize(self, args.transformer_init)

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
            logging.debug('loss (=%f) is not correct', loss.item())

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
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret

    def tokens_to_string(self, tokens):
        tokens = list(tokens)
        if self.eos in tokens:
            tokens = tokens[:tokens.index(self.eos)]
        return " ".join(self.char_list[token] for token in tokens if token != self.pad)