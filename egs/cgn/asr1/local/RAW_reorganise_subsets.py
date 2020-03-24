# coding: utf-8
import pandas as pd
import numpy as np
from pathlib import Path
import json
from IPython.display import display
import argparse

SPLITS = ("train", "dev", "test")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-length", default=0, type=int)
    parser.add_argument("--train-size", default=.6, type=float)
    parser.add_argument("--dev-size", default=.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--postfix", default="v0", type=str)
    parser.add_argument("--components", default=None, type=str, help="None or comma-separated letters")
    parser.add_argument("--datafiles", default="/users/spraak/qmeeus/spchdisk/data/cgn/datafiles.csv", type=Path)
    parser.add_argument("--train-dir", default="dump/train_m/deltafalse", type=Path)
    parser.add_argument("--save-utterances", action="store_true")
    return parser.parse_args()


def extract_uttids(jsonfile, minlength=0):
    with open(jsonfile) as infile:
        data = json.load(infile)["utts"]
    if minlength > 0:
        data = dict(filter(
            lambda item: item[1]["input"][0]["shape"][0] >= minlength, 
            data.items()
        ))

    assert len(data)
    return list(data)


def load_utterances(uttids, files, components=None):
    all_uttids = sum((uttids[key] for key in uttids), [])
    fileids = sum((
        list(map(lambda u: u.split("-")[1].split(".")[0], idlist)) 
        for subset, idlist in uttids.items()
    ), [])

    uttids2files = dict(zip(all_uttids, fileids))
    utterances = pd.DataFrame(uttids2files.items())
    assert utterances.shape[1] == 2
    utterances.columns = ["uttid", "name"]
    utterances = utterances.join(files.set_index("name")["comp"], on="name")

    if components is not None:
        components = [f"comp-{c}" for c in components.split(",")]
        utterances = utterances.loc[utterances["comp"].isin(components)]

    return utterances


def create_subsets(utterances, train_size, valid_size, random_state=None):
    utterances = utterances.sample(frac=1., random_state=random_state)
    train_size, dev_size = (int(len(utterances) * p) for p in (train_size, valid_size))
    test_size = len(utterances) - train_size - dev_size
    assert train_size + dev_size + test_size == len(utterances)
    subset_column_id = utterances.shape[1]
    utterances["subset"] = None
    utterances.iloc[:train_size, subset_column_id] = "train"
    utterances.iloc[train_size:train_size+dev_size, subset_column_id] = "dev"
    utterances.iloc[-test_size:, subset_column_id] = "test"
    utterances.loc[utterances["comp"] == "comp-n", "subset"] = "test"
    return utterances.sort_index()


def load_json(train_dir):
    files = list(Path(train_dir).glob("*.json"))
    from pprint import pprint
    print(f"Interactive cleaning of json files. {len(files)} files:")
    pprint(list(enumerate(files)))
    indices = list(range(len(files)))
    import ipdb; ipdb.set_trace()
    files = [files[i] for i in indices]
    jsonfiles = {split: [str(path).replace("train", split) for path in files] for split in SPLITS}
    assert all(Path(path).exists() for split in SPLITS for path in jsonfiles[split])
    return jsonfiles

def display_subset_distribution(utterances):
    display(utterances
            .groupby(["subset", "comp"])
            .count()["uttid"]
            .unstack(level=0)
            .fillna(0))

def reorganise_subsets(jsonfiles, subsets, postfix="v2"):
    """
    Create new jsonfiles from the original files with the new splits
    # jsonfiles = {"train": [path-to-json1, ...], "dev": ...}
    # subsets = {"train": [uttid1, ...], "dev": ...}
    """
    splits = list(jsonfiles)
    ndatasets = len(jsonfiles[splits[0]])
    for dataset_id in range(ndatasets):
        utterances = {}
        # Load and merge
        for split in splits:
            with open(jsonfiles[split][dataset_id]) as infile:
                utterances.update(json.load(infile)["utts"])
        # Split according to `subsets`
        newsplits = {split: {uttid: utterances[uttid] for uttid in subsets[split]} for split in splits}
        # Save new subsets in new json files
        for split in splits:
            filename = str(jsonfiles[split][dataset_id]).replace(".json", f".{postfix}.json")
            with open(filename, "w") as outfile:
                json.dump({"utts": newsplits[split]}, outfile)
            print(f"Written {filename} ({len(newsplits[split])} records)")


options = parse_args()
files = pd.read_csv(options.datafiles)
jsonfiles = load_json(options.train_dir)
uttids = {subset: extract_uttids(filelist[0], options.min_length) for subset, filelist in jsonfiles.items()}
fileids = {subset: sorted(set(map(lambda u: u.split("-")[1].split(".")[0], idlist))) for subset, idlist in uttids.items()}
valid_files = set(sum(fileids.values(), []))
files["is_valid"] = files["name"].isin(valid_files)
display(files[["comp", "name", "is_valid"]].groupby("comp").agg({"is_valid": "mean", "name": "count"}))

utterances = load_utterances(uttids, files, options.components)
utterances = create_subsets(utterances, options.train_size, options.dev_size, random_state=options.seed)

print(f"Total number of examples: {len(utterances)}")
display_subset_distribution(utterances)
print(f"New files will be labelled with {options.postfix}")
import ipdb; ipdb.set_trace()
display_subset_distribution(utterances)
print(f"Total number of examples: {len(utterances)} File postfix: {options.postfix}")

subsets = {
    split: utterances.loc[utterances["subset"] == split, "uttid"].tolist() 
    for split in SPLITS
}

reorganise_subsets(jsonfiles, subsets, options.postfix)

if options.save_utterances:
    utterances.to_csv("data/utterances.csv", index=False)


#%save -r local/RAW_reorganise_subsets.py 1-194
