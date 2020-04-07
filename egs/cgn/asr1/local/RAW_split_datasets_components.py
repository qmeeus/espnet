# coding: utf-8
import json
import pandas as pd


splits = pd.read_csv("data/CGN_ALL/splits.csv", index_col="uttid")
for split in ("train", "valid", "test"):

    with open(f"dump/CGN_{split}/deltafalse/data_unigram_1000.json") as f:
        subset = json.load(f)["utts"]

    groups = ["o", "ok", "ijklmno"]
    group_names = ["o", "ok", "mono"]

    for group_name, group in zip(group_names, groups):
        uttids = set(splits.loc[splits.comp.isin([f"comp-{x}" for x in list(group)])].index)
        selection = {uttid: utt for uttid, utt in subset.items() if uttid in uttids}
        filename = f"dump/CGN_{split}/deltafalse/data_unigram_1000.{group_name}.json"
        print(filename, len(selection))

        with open(filename, "w") as f:
            json.dump({"utts": selection}, f)

