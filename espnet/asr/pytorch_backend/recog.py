"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""
import os
import json
import logging
import numpy as np
import torch
import kaldiio
from editdistance import eval as editdistance
from operator import itemgetter
from itertools import groupby, product
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search import BeamSearch
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets


def recog(args):

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    device = "cuda" if args.ngpu == 1 else "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    float_type = dict(device=device, dtype=dtype)
    model.to(**float_type).eval()

    # read json data
    with open(args.recog_json, 'rb') as f:
        test_data = json.load(f)['utts']

    output_file = Path(args.result_label)
    att_dir = Path(output_file.parent, "att_ws")
    os.makedirs(att_dir, exist_ok=True)

    def clean_ctc_pred(ctc_pred):
        blank_id = model.char_list.index(model.blank)
        return list(filter(lambda x: x != blank_id, map(itemgetter(0), groupby(ctc_pred))))

    def clean_dec_pred(dec_pred):
        if model.eos in dec_pred:
            (eos_index, *_), = np.where(dec_pred == model.eos)
            return dec_pred[:eos_index]
        return dec_pred

    def tokens2str(y, sep=" "):
        return sep.join((model.char_list[yi] for yi in y))

    def tokens2sentence(y):
        return tokens2str(y, "").replace('▁', ' ')

    def accuracy(y_true, y_pred):
        return (y_true == y_pred).mean()

    def cer(y_true, y_pred):
        sent_true = tokens2str(y_true, "")
        sent_pred = tokens2str(y_pred, "")
        return editdistance(sent_true, sent_pred) / len(sent_true)

    def ter(y_true, y_pred):
        return editdistance(y_true, y_pred) / len(y_true)

    def plot_attention(att, uttid):
        sns.heatmap(att, cbar=False, cmap="viridis")
        plt.xlabel("Encoder index"); plt.ylabel("Decoder index")
        plt.tight_layout()
        plt.savefig(Path(att_dir, f"{uttid}.png"))
        plt.close()

    stats = OrderedDict({
        "id": [],
        "cer_ctc": [],
        "ter_ctc": [],
        "cer_dec": [],
        "ter_dec": [],
        "pred_ctc": [],
        "pred_dec": [],
        "groundtruth": []
    })

    with torch.no_grad():

        for idx, (uttid, sample) in enumerate(test_data.items(), 1):
            logging.info(f'({idx:d}/{len(test_data):d}) decoding {uttid}')
            xs = torch.as_tensor(kaldiio.load_mat(sample["input"][0]["feat"])).to(device).unsqueeze(0)
            xlens = torch.as_tensor([xs.size(1)]).to(device)
            tokens = np.fromiter(sample["output"][0]["tokenid"].split(), np.int)
            ys = torch.as_tensor(tokens).to(device).unsqueeze(0)
            hs_pad, hlens, ctc_out, dec_out, att, states = model._forward(xs, xlens, ys)
            ctc_prob = model.ctc.softmax(ctc_out)
            dec_prob = model.decoder.softmax(dec_out)
            ctc_pred = torch.argmax(ctc_prob, -1).cpu().detach().numpy()
            dec_pred = torch.argmax(dec_prob, -1).cpu().detach().numpy()

            ys = ys.cpu().detach().numpy()[0]
            ctc_pred = clean_ctc_pred(ctc_pred[0])
            dec_pred = clean_dec_pred(dec_pred)

            metrics = cer_ctc, ter_ctc, cer_dec, ter_dec = [
                scoring_func(ys, y_pred)
                for y_pred in (ctc_pred, dec_pred)
                for scoring_func in (cer, ter)
            ]

            sentences = ctc_sent, dec_sent, groundtruth = [
                tokens2sentence(y)
                for y in (ctc_pred, dec_pred, ys)
            ]

            logging.info(f"Ground truth: {groundtruth}")
            logging.info(f"Decoder prediction: {dec_sent}")
            logging.info(f"CTC prediction: {ctc_sent}")
            logging.info(
                f"[ Decoder CER {cer_dec:.2%} TER {ter_dec:.2%} ] [ CTC CER {cer_ctc:.2%} TER {ter_ctc:.2%} ]"
            )

            for key, value in zip(list(stats), [uttid] + metrics + sentences):
                stats[key] += [value]
                logging.debug(f"{key}: {value}")

            plot_attention(att[0], uttid)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': stats}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))


def recog_v2(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments. See py:func:`espnet.bin.asr_recog.get_parser` for details

    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False}
    )

    lm = None

    scorers = dict(
        **model.scorers(),
        lm=lm,
        length_bonus=LengthBonus(len(train_args.char_list))
    )

    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        length_bonus=args.penalty
    )

    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(train_args.char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=train_args.char_list,
    )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    elif args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"

    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()
    beam_search.to(device=device, dtype=dtype).eval()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            import ipdb; ipdb.set_trace()
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)[0][0]
            enc = model.encode(torch.as_tensor(feat).to(device=device, dtype=dtype))
            nbest_hyps = beam_search(x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), args.nbest)]]
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
