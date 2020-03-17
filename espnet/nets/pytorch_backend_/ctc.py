import logging

import numpy as np
from itertools import groupby
import editdistance

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import to_device


class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    :param str ctc_type: builtin or warpctc
    :param bool reduce: reduce the CTC loss into a scalar
    """

    def __init__(self, input_dim,
                 output_dim, 
                 dropout=.1, 
                 ctc_type='builtin', 
                 reduce=True, 
                 space_token="<space>", 
                 blank_token="<blank>", 
                 ignore_index=-1):

        super(CTC, self).__init__()

        if ctc_type != "builtin":
            raise ValueError(f'ctc_type must be "builtin": {ctc_type}')

        self.ctc_type = ctc_type
        self.space = space_token
        self.blank = blank_token
        self.ignore_id = ignore_index
        self.reduce = reduce
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.output_proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.ctc_loss = nn.CTCLoss(
            reduction='sum' if reduce else 'none', 
            zero_infinity=True
        )

    def loss_fn(self, outputs, target, output_lengths, target_lengths):
        logprobs = self.softmax(outputs)

        # Use the deterministic CuDNN implementation of CTC loss to avoid
        #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
        with torch.backends.cudnn.flags(deterministic=True):
            loss = self.ctc_loss(logprobs, target, output_lengths, target_lengths)

        # Batch-size average
        return loss / logprobs.size(1)

    def forward(self, hs_pad):
        # encoder output -> class probabilities
        logging.debug(f"ctc_input: {hs_pad.size()} fc_out: {self.output_proj.weight.size()}")
        return self.output_proj(self.dropout(hs_pad))

    def compute_loss(self, outputs, output_lengths, target, target_lengths):
        """
        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor target: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # ctc loss accepts both (B, Lo) and (B*sum(Lo)) targets
        # in the first case, targets are padded to the length of the longuest example
        # in the second case, targets are unpadded and concatenated in a 1d tensor 
        ys_true = target.flatten()  # parse padded ys
        ys_true = ys_true[ys_true != self.ignore_id].contiguous()

        # get ctc loss
        # expected shape of seqLength x batchSize x alphabet_size
        outputs = outputs.transpose(0, 1)

        loss = self.loss_fn(outputs, ys_true, output_lengths, target_lengths)
        if self.reduce:
            loss = loss.sum()
            logging.debug('ctc loss: {:.3f}'.format(loss.item()))

        return loss

    def compute_metrics(self, outputs, output_lengths, target, target_dict):

        predictions = torch.argmax(outputs, dim=-1)

        # 1. compute cer without beam search

        map_func = lambda i: target_dict[i]
        ignore_id = lambda i: i != self.ignore_id
        ignore_blank = lambda t: t != self.blank

        cer_scores = torch.zeros(*outputs.size()[:-1]).type_as(outputs)
        for i, y_pred in enumerate(predictions):
            y_pred = torch.unique_consecutive(y_pred[0])
            y_true = target[i]

            seq_pred, seq_true = (
                map(list, (                                 # Output is a list
                    filter(ignore_blank, map(               # Remove blank tokens
                        map_func, filter(ignore_id, y)      # Remove padding and convert to target
                    )) for y in (y_pred, y_true)
                ))
            )

            if seq_true:
                cer_scores[i] = editdistance.eval(seq_pred, seq_true) / len(seq_true)

        cer_ctc = cer_scores.mean()

        # # 2. compute cer/wer

        # if self.training or not (self.report_cer or self.report_wer):
        #     cer, wer = torch.zeros(2).type_as(outputs)

        # else:
        #     lpz = self.softmax(outputs) if self.recog_args.ctc_weight > 0.0 else None

        #     word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
        #     nbest_hyps = self.decoder.recognize_beam_batch(
        #         outputs, output_lengths, lpz,
        #         self.recog_args, self.char_list,
        #         self.rnnlm)
        #     # remove <sos> and <eos>
        #     y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
        #     for i, y_hat in enumerate(y_hats):
        #         y_true = target[i]

        #         seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
        #         seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
        #         seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
        #         seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
        #         seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

        #         hyp_words = seq_hat_text.split()
        #         ref_words = seq_true_text.split()
        #         word_eds.append(editdistance.eval(hyp_words, ref_words))
        #         word_ref_lens.append(len(ref_words))
        #         hyp_chars = seq_hat_text.replace(' ', '')
        #         ref_chars = seq_true_text.replace(' ', '')
        #         char_eds.append(editdistance.eval(hyp_chars, ref_chars))
        #         char_ref_lens.append(len(ref_chars))

        #     wer = float(sum(word_eds)) / sum(word_ref_lens)
        #     cer = float(sum(char_eds)) / sum(char_ref_lens)

        # # HACK
        # wer, cer, cer_ctc = (
        #     torch.tensor(m).type_as(outputs) for m in (wer, cer, cer_ctc)
        # )

        cer, wer = torch.zeros(2).type_as(outputs)

        return cer_ctc, cer, wer


def ctc_for(args, output_dim, reduce=True):
    """Returns the CTC module for the given args and output dimension

    :param Namespace args: the program args
    :param int output_dim : The output dimension
    :param bool reduce : return the CTC loss in a scalar
    :return: the corresponding CTC module
    """
    num_encs = getattr(args, "num_encs", 1)  # use getattr to keep compatibility
    
    if num_encs < 1:
        raise ValueError(f"Invalid argument for num_encs: {num_encs}")

    options = dict(
        input_dim=args.eprojs,
        output_dim=output_dim,
        ctc_type=args.ctc_type, 
        reduce=reduce, 
        space_token=args.sym_space, 
        blank_token=args.sym_blank, 
    )

    if num_encs == 1:
        # compatible with single encoder asr mode
        return CTC(**options, dropout=args.dropout_rate)
    
    if args.share_ctc:
        # use dropout_rate of the first encoder
        return torch.nn.ModuleList([CTC(**options, dropout=args.dropout_rate[0])])

    return torch.nn.ModuleList([
        CTC(**options, dropout=args.dropout_rate[idx])
        for idx in range(num_encs)
    ])
