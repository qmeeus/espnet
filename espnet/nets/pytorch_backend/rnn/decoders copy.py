from distutils.version import LooseVersion
import logging
import math
import random
import six

import numpy as np
import torch
import torch.nn.functional as F

from argparse import Namespace

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.utils.torch_utils import load_pretrained_embedding_from_file

MAX_DECODER_OUTPUT = 5
CTC_SCORING_RATIO = 1.5

# EMB_PATH = "/esat/spchdisk/scratch/qmeeus/data/dutchembeddings/160/combined-160.txt"
EMB_PATH = "/esat/spchdisk/scratch/qmeeus/data/cgn/preprocessed/w2v_small.txt"


class Decoder(torch.nn.Module):
    """Decoder module

    :param int eprojs: encoder projection units
    :param int odim: dimension of outputs
    :param str dtype: gru or lstm
    :param int dlayers: decoder layers
    :param int dunits: decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    :param float sampling_probability: scheduled sampling probability
    :param float dropout: dropout rate
    :param float context_residual: if True, use context vector for token generation
    :param float replace_sos: use for multilingual (speech/text) translation
    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0,
                 dropout=0.0, context_residual=False, replace_sos=False, num_encs=1):

        torch.nn.Module.__init__(self)
        self.rnn_type = dtype
        self.dunits = dunits
        self.num_layers = dlayers
        self.context_residual = context_residual
        # self.embed = torch.nn.Embedding(odim, dunits)
        emb_dim = len(char_list) - 1
        self.embed = load_pretrained_embedding_from_file(
            EMB_PATH, char_list, freeze=True, special_indices=[1, 0, -1]
        )

        self.dropout_emb = torch.nn.Dropout(p=dropout)

        RNN = torch.nn.LSTM if self.rnn_type == "lstm" else torch.nn.GRU
        self.rnn = RNN(
            self.embed.embedding_dim + eprojs, 
            dunits, 
            num_layers=dlayers, 
            bidirectional=False,
            dropout=dropout, 
            batch_first=False
        )

        self.ignore_id = -1

        self.output = torch.nn.Linear(
            (dunits + eprojs) if context_residual else dunits, 
            self.embed.embedding_dim
        )

        self.attention = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list

        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual E2E-ST
        self.replace_sos = replace_sos

        self.logzero = -1e10

    def init_weights(self):
        # embed weight ~ Normal(0, 1)
        self.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for layer in range(self.num_layers):
            bias_ih = getattr(self.rnn, f"bias_ih_l{layer}")
            # bias_ih = self.rnn[layer].bias_ih
            set_forget_bias_to_one(bias_ih)

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, enc_out, hidden_states):
        bs, input_size = enc_out.size()
        enc_out = enc_out.view(1, bs, input_size)
        dec_out, hidden_states = self.rnn(enc_out, hidden_states)

        # HACK: Compat
        if self.rnn_type == "gru":
            return hidden_states, None

        return hidden_states
        
    def forward(self, hs_pad, hlens, prev_output_tokens, strm_idx=0):
        """Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
                                    [in multi-encoder case,
                                    list of torch.Tensor, [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
                                    [in multi-encoder case, list of torch.Tensor, [(B), (B), ..., ]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index indicates the index of decoding stream.
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        """
        # to support mutiple encoder asr mode, in single encoder mode, convert torch.Tensor to List of torch.Tensor
        batch_size, max_output_length = prev_output_tokens.size()
        
        # initialization
        hidden_states = torch.stack([self.zero_state(hs_pad)] * self.num_layers)
        if self.rnn_type == "lstm":
            cell_states = torch.stack([self.zero_state(hs_pad)] * self.num_layers)
        else:
            cell_states = None

        # attention index for the attention module
        # in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = 0

        # hlens should be list of list of integer --> TODO: WHY??
        hlens = list(map(int, hlens))

        attention_weights = []
        decoder_outputs = []
        att_w = None
        self.attention[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        prev_tokens_emb = self.dropout_emb(self.embed(prev_output_tokens))  # utt x olen x zdim

        # loop for an output sequence
        for i in range(max_output_length):
            
            att_c, att_w = self.attention[att_idx](hs_pad, hlens, hidden_states[0], att_w)
            
            if i > 0 and random.random() < self.sampling_probability:
                prev_pred = self._sample_from_previous_output(decoder_outputs[-1])
            else:
                prev_pred = prev_tokens_emb[:, i, :]

            ey = torch.cat((prev_pred, att_c), dim=1)  # utt x (zdim + hdim)
            hidden_states, cell_states = self.rnn_forward(ey, (hidden_states, cell_states))

            dec_out = hidden_states[-1]  # utt x (zdim)
            if self.context_residual:
                dec_out = torch.cat([dec_out, att_c], dim=-1) # utt x (zdim + hdim)

            attention_weights.append(att_w)
            decoder_outputs.append(dec_out)

        decoder_outputs = torch.stack(decoder_outputs, dim=1).view(batch_size * max_output_length, -1)
        return self.output(decoder_outputs), attention_weights

    def compute_loss(self, hs_pad, hlens, ys_pad, strm_idx=0):
        batch_size, max_output_length = ys_pad.size()
        max_output_length += 1

        # FIXME
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        average_length = np.mean(list(map(len, ys)))
        prev_output_tokens = self._add_sos_token(ys, pad=True, pad_value=self.eos)
        eos = ys_pad[0].new([self.eos])
        target = [torch.cat([y, eos], dim=0) for y in ys]
        y_true_pad = pad_list(target, self.ignore_id)

        # get dim, length info
        logging.debug(f"input lengths: {hlens[0]}.")
        logging.debug(f"{self.__class__.__name__} output lengths: {[len(y) for y in target]}")

        predictions, _ = self.forward(hs_pad, hlens, prev_output_tokens)

        loss = F.cross_entropy(predictions, y_true_pad.view(-1),
                               ignore_index=self.ignore_id,
                               reduction="mean")


    def _sample_from_previous_output(self, prediction):
        logging.debug(' scheduled sampling ')
        z_out = self.output(prediction)
        z_out = np.argmax(z_out.detach().cpu(), axis=1)
        z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
        return z_out

    def _add_sos_token(self, target, pad=True, pad_value=0):
        """Add sos token to target to be used as the decoder input
        :param list of torch.Tensor target: output tokens
        :param bool pad: return padded tokens
        :param pad_value int: the value to be used for padding
        :rtype: list of tensors
        :return: the previous output tokens, padded with pad_value if specified
        """
        sos = target[0].new([self.sos])
        prev_output_tokens = [torch.cat([sos, y], dim=0) for y in target]
        if pad:
            prev_output_tokens = pad_list(prev_output_tokens, pad_value)
        return prev_output_tokens

    def calculate_all_attentions(self, hs_pad, hlens, ys_pad, strm_idx=0):
        """Calculate all of attentions

            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
                                        [in multi-encoder case,
                                        list of torch.Tensor, [(B, Tmax_1, D), (B, Tmax_2, D), ..., ] ]
            :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
                                        [in multi-encoder case, list of torch.Tensor, [(B), (B), ..., ]
            :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
            :param int strm_idx: stream index for parallel speaker attention in multi-speaker case
            :param torch.Tensor lang_ids: batch of target language id tensor (B, 1)
            :return: attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) multi-encoder case => [(B, Lmax, Tmax1), (B, Lmax, Tmax2), ..., (B, Lmax, NumEncs)]
                3) other case => attention weights (B, Lmax, Tmax).
            :rtype: float ndarray
        """

        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # hlen should be list of list of integer
        prev_output_tokens = self._add_sos_token(ys, pad=True, pad_value=self.eos)
        _, att_ws = self.forward(hs_pad, hlens, prev_output_tokens)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        return att_to_numpy(att_ws, self.attention[0])

def decoder_for(args, odim, sos, eos, att, labeldist):
    return Decoder(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, sos, eos, att, args.verbose,
                   args.char_list, labeldist,
                   args.lsm_weight, args.sampling_probability, args.decoder_dropout,
                   getattr(args, "context_residual", False),  # use getattr to keep compatibility
                   getattr(args, "replace_sos", False),  # use getattr to keep compatibility
                   getattr(args, "num_encs", 1))  # use getattr to keep compatibility
