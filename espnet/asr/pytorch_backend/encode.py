import json
import logging
from configargparse import Namespace
import torch
import os
import numpy as np
import pandas as pd
import re
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
import torch
from tqdm import tqdm
from editdistance import eval as editdistance
from itertools import groupby

from torch.nn.utils.rnn import pad_sequence

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.batchfy import make_batchset
from espnet.asr.pytorch_backend.asr_init import load_trained_modules, load_trained_model
from espnet.utils.dynamic_import import dynamic_import
from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.io_utils import LoadInputsAndTargets
# from espnet.utils.torch_utils import _recursive_to
from espnet.nets.pytorch_backend.nets_utils import pad_list


def encode(options):

    options.batchsize = 1
    model, dataset, loader, options, train_args, device, dtype = setup(options)

    if "maskctc" in str(model.__class__):
        _encode_maskctc(model, dataset, loader, options, train_args, device, dtype)

    else:
        _encode(model, dataset, loader, options, train_args, device, dtype)


def setup(options):

    # SETUP AND PRELIMINARY OPERATIONS
    set_deterministic_pytorch(options)

    # check the use of multi-gpu
    if options.ngpu > 1:
        if options.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                options.batch_size, options.batch_size * options.ngpu))
            options.batch_size *= options.ngpu
        if options.num_encs > 1:
            # TODO(ruizhili): implement data parallel for multi-encoder setup.
            raise NotImplementedError("Data parallel is not supported for multi-encoder setup.")

    model, train_args = load_trained_model(options.model)
    model.eval()
    if hasattr(model, "teacher_model"):
        del model.teacher_model
    for param in model.parameters():
        param.requires_grad = False

    with open(options.recog_json, "rb") as json_file:
        metadata = json.load(json_file)["utts"]

    dataset = make_batchset(
        metadata, options.batchsize,
    )

    loader = LoadInputsAndTargets(
        mode='asr',
        load_output=True,
        preprocess_conf=options.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    if options.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if options.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    
    dtype = getattr(torch, options.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()

    os.makedirs(options.outdir, exist_ok=True)

    return model, dataset, loader, options, train_args, device, dtype


def _encode(model, dataset, loader, options, train_args, device, dtype):

    h5f = h5py.File(f"{options.outdir}/{options.outfile}.h5", "w")
    output = {}
    np.seterr(divide="ignore")

    with torch.no_grad():

        for batch in tqdm(dataset):

            uttids = [example[0] for example in batch]
            batch_x, batch_y = loader(batch)
            x_lens = torch.tensor(list(map(len, batch_x)))
            batch_x = pad_sequence(
                list(map(torch.tensor, batch_x)), batch_first=True, padding_value=0.
            )

            blank_multiplier = 1
            dec_lprobs = torch.tensor([])
            while not dec_lprobs.numel() and blank_multiplier > .2:
                dec_lprobs, ctc_preds, encoded, lengths = model.predict(
                    batch_x.to(device), x_lens, K=0, p_thres=.9, blank_multiplier=blank_multiplier
                )
                blank_multiplier /= 2
            
            encoded, lengths = (tensor.cpu().numpy() for tensor in (encoded, lengths))

            ctc_tokens = [list(map(train_args.char_list.__getitem__, yhat)) for yhat in ctc_preds] 
            dec_preds = [yhat[1:olen] for yhat, olen in zip(dec_lprobs.argmax(-1), lengths)]
            dec_tokens = [model.teacher_tokenizer.convert_ids_to_tokens(yhat, skip_special_tokens=True) for yhat in dec_preds]

            for (uttid, sample), y_ctc, y_dec, x_enc, x_len in zip(batch, ctc_tokens, dec_tokens, encoded, lengths):
                s_ctc = "".join(y_ctc).replace("▁", " ")
                s_dec = model.teacher_tokenizer.convert_tokens_to_string(y_dec)
                s_true = sample["output"][0]["text"]
                output[uttid] = {
                    "ctc": {
                        "tokens": y_ctc, "text": s_ctc, 
                        "score": {"wer": wer(s_ctc, s_true), "cer": cer(s_ctc, s_true)}
                    },
                    "decoder": {
                        "tokens": y_dec, "text": s_dec, 
                        "score": {"wer": wer(s_dec, s_true), "cer": cer(s_dec, s_true)}
                    },
                    "target": {"tokens": sample["output"][0]["token"], "text": s_true},
                }

                h5f.create_dataset(
                    uttid, 
                    data=x_enc[:x_len], 
                    compression="gzip", 
                    compression_opts=9
                )

    h5f.close()

    with open(f"{options.outdir}/{options.outfile}.json", "w") as jsonfile:
        json.dump({"results": output}, jsonfile, indent=4, sort_keys=True)

    results = pd.DataFrame.from_dict({
        uttid: {
            f"{output_type}_{metric}": value[output_type]["score"][metric] 
            for output_type in ["ctc", "decoder"] 
            for metric in ["wer", "cer"]
        } for uttid, value in output.items()
    }, orient="index")
    
    results["speaker"] = results.index.map(lambda s: re.split("_|-", s)[0])
    results.groupby("speaker").agg(["mean", "std"]).to_csv(f"{options.outdir}/{options.outfile}.csv")


def _encode_maskctc(model, dataset, loader, options, train_args, device, dtype):

    if options.batchsize != 1:
        raise TypeError("Batch size must be one with maskctc")

    h5f = h5py.File(f"{options.outdir}/{options.outfile}.h5", "w")
    output = {}
    np.seterr(divide="ignore")

    recog_args = Namespace()
    recog_args.maskctc_probability_threshold = .95
    recog_args.maskctc_n_iterations = 10

    model = model.to("cpu")

    ids2tokens = lambda ids: list(map(train_args.char_list.__getitem__, ids))
    tokens2text = "".join

    with torch.no_grad():

        for batch in tqdm(dataset):

            uttid, = [example[0] for example in batch]
            batch_x, batch_y = loader(batch)
            x_lens = torch.tensor(list(map(len, batch_x)))
            batch_x = pad_sequence(
                list(map(torch.tensor, batch_x)), batch_first=True, padding_value=0.
            )

            assert batch_x.size(0) == 1
            batch_x = batch_x.squeeze(0).to("cpu")

            hyp, = model.recognize(batch_x, recog_args, char_list=train_args.char_list)
            dec_ids = torch.tensor(hyp["yseq"])
            h_enc = model.encode(batch_x)
            _, ctc_raw = torch.exp(model.ctc.log_softmax(h_enc.unsqueeze(0)).squeeze(0)).max(dim=-1)
            ctc_preds = torch.stack([x[0] for x in groupby(ctc_raw)])
            ctc_ids = ctc_preds[torch.nonzero(ctc_preds != 0).squeeze(-1)]
            _, _, h_dec = model.decoder(dec_ids.unsqueeze(0), None, h_enc, None, return_hidden=True)

            gold_ids = batch_y[0][:-1].tolist()
            ctc_ids = ctc_ids.tolist()
            dec_ids = dec_ids[1:-1].tolist()
            tokens = [gold_tokens, ctc_tokens, dec_tokens] = list(map(ids2tokens, [gold_ids, ctc_ids, dec_ids]))
            gold_text, ctc_text, dec_text = map(tokens2text, tokens)

            output[uttid] = {
                "ctc": {
                    "tokens": ctc_ids, "text": ctc_text,
                    "score": {"wer": wer(ctc_text, gold_text), "cer": cer(ctc_text, gold_text)}
                },
                "decoder": {
                    "tokens": dec_ids, "text": dec_text,
                    "score": {"wer": wer(dec_text, gold_text), "cer": cer(dec_text, gold_text)}
                },
                "target": {
                    "tokens": list(gold_ids), "text": gold_text
                }            
            }

            h5f.create_dataset(
                uttid, 
                data=h_dec.squeeze(0), 
                compression="gzip", 
                compression_opts=9
            )

    h5f.close()

    with open(f"{options.outdir}/{options.outfile}.json", "w") as jsonfile:
        json.dump({"results": output}, jsonfile, indent=4, sort_keys=True)

    results = pd.DataFrame.from_dict({
        uttid: {
            f"{output_type}_{metric}": value[output_type]["score"][metric] 
            for output_type in ["ctc", "decoder"] 
            for metric in ["wer", "cer"]
        } for uttid, value in output.items()
    }, orient="index")
    
    results["speaker"] = results.index.map(lambda s: re.split("_|-", s)[0])
    results.groupby("speaker").agg(["mean", "std"]).to_csv(f"{options.outdir}/{options.outfile}.csv")



def wer(pred, target):
    pred, target = (seq.split() for seq in (pred, target))
    return np.divide(editdistance(pred, target), len(target))
    

def cer(pred, target):
    pred, target = (seq.replace(" ", "") for seq in (pred, target))
    return np.divide(editdistance(pred, target), len(target))