#!/usr/bin/env python3
import json
import kaldiio
import pandas as pd
import multiprocessing as mp

targets = ["unigram_1000", "pos"]
data = "data"
dump = "dump"
prefix = "CGN"
subsets = ["train", "valid", "test"]
splits = pd.read_csv(f"{data}/CGN_ALL/splits.csv", index_col="uttid")


def get_shape(feats):
    mat = kaldiio.load_mat(feats)
    return mat.shape


def format_features(sample):
    return {
        "feat": sample["features"],
        "name": "fbanks",
        "shape": get_shape(sample["features"])
    }


def serialise(uttid, sample):
    return uttid, {
        "input": [format_features(sample)],
        "output": [create_target(sample, target) for target in targets],
        "utt2spk": sample["speaker"],
        "lang": sample["lang"],
        "comp": sample["comp"]
    }


def create_json(subset):
    dataset = splits[splits[subset]].copy()
    with open(f"{dump}/{prefix}_{subset}/data.json", "w") as f:
        json.dump({"utts": dict(zip(*dataset.apply(serialise, axis=1)))})


#for subset in subsets:
#    p = mp.Process(target=create_json, args=(subset,))


