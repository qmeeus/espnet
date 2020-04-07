#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from IPython.display import display


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--train-size", type=float, default=.8)
    parser.add_argument("--valid-size", type=float, default=.05)
    parser.add_argument("--test-size", type=float, default=.15)
    parser.add_argument("--comp-for-test", type=str, default=None,
                        help="comma-separated letters corresponding to components in CGN")
    return parser.parse_args()


args = parse_args()
subsets = ["train", "valid", "test"]

feats = (
    pd.read_csv(Path(args.data, "feats.scp"),
                header=None,
                names=["uttid", "features"],
                sep=" ",
                index_col="uttid")
    .join(pd.read_csv(Path(args.data, "annotations.csv"), index_col="uttid"),how='left')
)

for subset in subsets:
    feats[subset] = False

if args.comp_for_test is not None:
    test_comps = list(map("comp-{}".format, map(str.strip, args.comp_for_test.split(","))))
    feats.loc[feats["comp"].isin(test_comps), "test"] = True

test_size = int(len(feats) * args.test_size) - feats["test"].sum()
valid_size = int(len(feats) * args.valid_size)

feats.loc[feats[~feats["test"]].sample(test_size).index, "test"] = True
feats.loc[feats[~feats["test"]].sample(valid_size).index, "valid"] = True
feats.loc[~feats["test"] & ~feats["valid"], "train"] = True

splits_summary = feats.groupby("comp").agg(dict(features="count", **{subset: "sum" for subset in subsets}))
splits_summary["train (%)"], splits_summary["valid (%)"], splits_summary["test (%)"] = \
    zip(*splits_summary.apply(lambda row: [row[subset] / row["features"] for subset in subsets], axis=1))
display(splits_summary)

feats.reset_index().to_csv(Path(args.data, "splits.csv"), index=False)

for subset in subsets:
    savedir = (args.prefix or args.data.name) + f"_{subset}"
    savedir = Path(args.data.parent, savedir)
    os.makedirs(savedir, exist_ok=True)
    uttids = set(feats.loc[feats[subset]].index)
    cond = lambda line: line.split()[0] in uttids
    for filename in ["feats.scp", "segments", "text", "utt2spk", "utt2dur", "utt2num_frames"]:
        with open(Path(args.data, filename)) as infile, open(Path(savedir, filename), "w") as outfile:
            outfile.write("".join(filter(cond, infile.readlines())))
            print(f"{savedir}/{filename} created!")

