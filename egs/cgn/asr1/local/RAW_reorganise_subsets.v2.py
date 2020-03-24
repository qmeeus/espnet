# coding: utf-8
def get_annotation_files(tag_dir, comps, lang="vl"):
        tag_files = []
            for comp in comps:
                        tag_files.extend(list(tag_dir.glob(f"comp-{comp}/{lang}/*.tag.gz")))
    tag_files = dict(map(lambda v: (v.stem.replace(".tag", ""), v), tag_files))
ls /users/spraak/qmeeus/spchdisk/data/cgn/
!head /users/spraak/qmeeus/spchdisk/data/cgn/datafiles.csv
import pandas as pd
files = pd.read_csv("/users/spraak/qmeeus/spchdisk/data/cgn/datafiles.csv")
files
train, dev, test = (f"dump/{subset}_m" for subset in ("train", "dev", "test"))
from pathlib import Path
datapaths = dict(zip(("train", "dev", "test"), (f"dump/{subset}_m" for subset in ("train", "dev", "test"))))
datapaths
datapaths = dict(zip(("train", "dev", "test"), (f"dump/{subset}_m/deltafalse" for subset in ("train", "dev", "test"))))
datapaths
jsonfiles = {subset: Path(path).glob("*.json") for subset, path in datapaths.items()}
jsonfiles
jsonfiles = {subset: list(Path(path).glob("*.json")) for subset, path in datapaths.items()}
jsonfiles
def extract_uttids(jsonfile):
    with open(jsonfile) as infile:
        return list(json.load(jsonfile)["utts"])
        
import json
uttids = {subset: extract_uttids(filelist[0]) for subset, filelist in jsonfiles.items()}
def extract_uttids(jsonfile):
    with open(jsonfile) as infile:
        return list(json.load(infile)["utts"])
        
uttids = {subset: extract_uttids(filelist[0]) for subset, filelist in jsonfiles.items()}
uttids["train"][:10]
fileids = {subset: uttid.split("-")[1].split(".")[0] for subset, uttid in uttids.items()}
fileids = {subset: list(map(lambda u: u.split("-")[1].split(".")[0], idlist)) for subset, idlist in uttids.items()}
fileids["train"][:20]
fileids = {subset: sorted(set(map(lambda u: u.split("-")[1].split(".")[0], idlist))) for subset, idlist in uttids.items()}
fileids["train"][:20]
from itertools import zip_longest
files
subsets = pd.DataFrame([{v: k} for v in values for k, v in fileids.items()])
subsets = pd.DataFrame([{id: subset} for id in values for subset, ids in fileids.items()])
subsets = pd.DataFrame([{id: subset} for id in ids for subset, ids in fileids.items()])
subsets = pd.DataFrame([{id: subset} for subset, ids in fileids.items() for id in ids])
subsets
subsets = pd.DataFrame(sum(fileids.values(), []))
subsets
subsets.columns = ["name"]
subsets["subset"] = None
subsets.loc[0:len(fileids["train"]), "subset"] = "train"
subsets.loc[len(fileids["train"]):len(fileids["train"])+len(fileids["dev"]), "subset"] = "dev"
subsets["subset"].isnull().sum()
len(fileids["test"])
fileids["train"][-1]
subsets[len(fileids["train"])]
subsets.loc[len(fileids["train"])]
fileids["train"][-1]
subsets[len(fileids["train"]) + 1]
subsets.loc[len(fileids["train"])]
fileids["train"][-1]
subsets.loc[len(fileids["train"]) + 1]
subsets.loc[len(fileids["train"]) + 2]
subsets.loc[subsets["name"] == "fv801488"]
subsets.shape
subsets = pd.DataFrame()
subsets["name"] = fileids["train"]
files["subset"] = None
files
train_size, dev_size, test_size = .7, .1, .2
from sklearn.model_selection import train_test_split
valid_files = sum(fileids.values(), [])
len(valid__files)
len(valid_files)
len(fileids["train"])
len(fileids["train"]) + len(fileids["dev"])
len(fileids["train"]) + len(fileids["dev"]) + len(fileids["test"])
valid_files = set(valid_files)
files["is_valid"] = files["name"].isin(valid_files)
files
files[files["is_valid"]]
files.loc[files["is_valid"], ["comp", "name"]].groupby("comp").count()
files.loc[1, "audio"]
ls /users/spraak/qmeeus/data/cgn/
files.loc[files["is_valid"], ["comp", "name"]].groupby("lang").count()
files
files.loc[files["is_valid"], ["comp", "name", "lang"]].groupby("lang").count()
files.loc[files["is_valid"], ["comp", "name"]].groupby("comp").count()
locals().keys()
jsonfiles
uttids["train"][:10]
[k for k in locals().keys() if not k.startswith("_")]
fileids[:10]
fileids
uttids[:10]
uttids["train"][:10]
all_uttids = sum(uttids[key] for key in uttids, [])
all_uttids = sum((uttids[key] for key in uttids), [])
fileids["train"][:10]
fileids = sum((list(map(lambda u: u.split("-")[1].split(".")[0], idlist)) for subset, idlist in uttids.items()), [])
uttids2files = dict(zip(uttids, fileids))
uttids2files.keys()
fileids["train"][:10]
fileids["train"]
uttids2files["train"][:10]
fileids[:10]
all_uttids[:10]
uttids2files = dict(zip(all_uttids, fileids))
uttids2files[:10]
uttids2files["V40244-fv400782.35""]
uttids2files["V40244-fv400782.35"]
import numpy as np
np.random.shuffle(all_uttids)
train_size, dev_size, test_size = (len(all_uttids) * p for p in (.6, .1, .3))
train_size, dev_size, test_size = (int(len(all_uttids) * p) for p in (.6, .1, .3))
train_size, dev_size, test_size
train_ids, dev_ids, test_ids = uttids2files[:train_size], uttids2files[train_size:train_size+dev_size], uttids2files[-test_size:]
train_ids, dev_ids, test_ids = all_uttids[:train_size], all_uttids[train_size:train_size+dev_size], all_uttids[-test_size:]
len(train_ids) + len(dev_ids) + len(test_ids)
len(all_uttids)
train_size + dev_size + test_size
train_size, dev_size = (int(len(all_uttids) * p) for p in (.6, .1))
test_size = len(all_uttids) - train_size - dev_size
train_ids, dev_ids, test_ids = all_uttids[:train_size], all_uttids[train_size:train_size+dev_size], all_uttids[-test_size:]
train_size + dev_size + test_size
len(all_uttids)
files
utterances = pd.DataFrame(uttids2files)
utterances = pd.DataFrame(uttids2files.items())
utterances
utterances.columns = ["uttid", "name"]
utterances.shuffle()
utterances = utterances.sample(1.)
utterances.sample?
utterances = utterances.sample(frac=1.)
utterances["subset"] = None
utterances
utterances.iloc[:train_size, "subset"] = "train"
utterances
utterances.iloc[:train_size, 2] = "train"
utterances.iloc[train_size:train_size+dev_size, 2] = "dev"
utterances.iloc[-test_size:, 2] = "test"
utterances = utterances.sort_index()
utterances
files
utterances.join?
utterances = utterances.join(files.set_index("name")["comp"], on="name")
utterances
utterances.groupby(["subset", "comp"]).count()
utterances.groupby(["subset", "comp"]).count()["uttid"].unstack(level=0)
utterances.groupby(["subset", "comp"]).count()["uttid"].unstack(level=0).mean(0)
subset_distribution = utterances.groupby(["subset", "comp"]).count()["uttid"].unstack(level=0)
subset_distribution / subset_distribution.sum(1)
subset_distribution / subset_distribution.sum(0)
(subset_distribution / subset_distribution.sum(0)).sum(1)
subset_distribution / subset_distribution.sum(0)
(subset_distribution.T / subset_distribution.sum(1)).T
subset_distribution
subset_distribution.sum(1)
subset_distribution.mean(1)
subset_distribution.sum(1)
subset_distribution.sum(1).plot.bar()
import matplotlib.pyplot as plt
plt.show()
plt.close()
plt.close()
subset_distribution.sum(1) / subset_distribution.sum()
subset_distribution.sum()
subset_distribution.sum(1) / subset_distribution.sum().sum()
subset_distribution.sum(1)
utterances[utterances["comp"] = "comp-n"]
utterances[utterances["comp"] == "comp-n"]
utterances.loc[utterances["comp"] == "comp-n", "subset"] = "test"
subset_distribution = utterances.groupby(["subset", "comp"]).count()["uttid"].unstack(level=0)
subset_distribution
subset_distribution = utterances.groupby(["subset", "comp"]).count()["uttid"].unstack(level=0).fillna(0)
subset_distribution
subset_distribution.sum() / subset_distribution.sum([0,1])
subset_distribution.sum?
subset_distribution.sum() / subset_distribution.sum().sum()
def reorganise_subsets(jsonfiles, subsets):
    splits = list(jsonfiles)
    ndatasets = len(jsonfiles[splits[0]])
    for dataset_id in range(ndatasets):
        utterances = {}
        # Load and merge
        for split in splits:
            with open(jsonfiles[split][dataset_id]) as infile:
                utterances.update(json.load(infile)["utts"])
        # Split according to `subsets`
        newsplits = [{uttid: utterances[uttid] for uttid in subsets[split]} for split in splits]
        # Save new subsets in new json files
        for split in splits:
            filename = jsonfiles[split][dataset_id].repace(".json", ".v2.json")
            with open(filename, "w") as outfile:
                json.dump(newsplits[split], outfile)
            print(f"Written {filename} ({len(newsplits[split])} records)")
            
import json
def reorganise_subsets(jsonfiles, subsets):
    # jsonfiles = {"train": [path-to-json1, ...], "dev": ...}
    # subsets = {"train": [uttid1, ...], "dev": ...}
    splits = list(jsonfiles)
    ndatasets = len(jsonfiles[splits[0]])
    for dataset_id in range(ndatasets):
        utterances = {}
        # Load and merge
        for split in splits:
            with open(jsonfiles[split][dataset_id]) as infile:
                utterances.update(json.load(infile)["utts"])
        # Split according to `subsets`
        newsplits = [{uttid: utterances[uttid] for uttid in subsets[split]} for split in splits]
        # Save new subsets in new json files
        for split in splits:
            filename = jsonfiles[split][dataset_id].repace(".json", ".v2.json")
            with open(filename, "w") as outfile:
                json.dump(newsplits[split], outfile)
            print(f"Written {filename} ({len(newsplits[split])} records)")
            
jsonfiles
print([len(jf) for split in jsonfiles for jf in jsonfiles[split]])
print([len(jsonfiles[split]) for split in jsonfiles])
jsonfiles["train"].pop(5)
jsonfiles["dev"].pop(5)
jsonfiles
print([len(jsonfiles[split]) for split in jsonfiles])
jsonfiles["dev"].pop(6)
print([len(jsonfiles[split]) for split in jsonfiles])
utterances
subsets
subsets = {split: utterances.loc[utterances["subset"] == split, "uttid"].tolist() for split in ("train", "dev", "test")}
subsets
reorganise_subsets(jsonfiles, subsets)
def reorganise_subsets(jsonfiles, subsets):
    # jsonfiles = {"train": [path-to-json1, ...], "dev": ...}
    # subsets = {"train": [uttid1, ...], "dev": ...}
    splits = list(jsonfiles)
    ndatasets = len(jsonfiles[splits[0]])
    for dataset_id in range(ndatasets):
        utterances = {}
        # Load and merge
        for split in splits:
            with open(jsonfiles[split][dataset_id]) as infile:
                utterances.update(json.load(infile)["utts"])
        # Split according to `subsets`
        newsplits = [{uttid: utterances[uttid] for uttid in subsets[split]} for split in splits]
        # Save new subsets in new json files
        for split in splits:
            filename = str(jsonfiles[split][dataset_id]).replace(".json", ".v2.json")
            with open(filename, "w") as outfile:
                json.dump(newsplits[split], outfile)
            print(f"Written {filename} ({len(newsplits[split])} records)")
            
reorganise_subsets(jsonfiles, subsets)
def reorganise_subsets(jsonfiles, subsets):
    # jsonfiles = {"train": [path-to-json1, ...], "dev": ...}
    # subsets = {"train": [uttid1, ...], "dev": ...}
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
            filename = str(jsonfiles[split][dataset_id]).replace(".json", ".v2.json")
            with open(filename, "w") as outfile:
                json.dump({"utts": newsplits[split]}, outfile)
            print(f"Written {filename} ({len(newsplits[split])} records)")
            
            
reorganise_subsets(jsonfiles, subsets)
utterances
utterances.to_csv("data/utterances.csv", index=False)
files
%save -r local/RAW_reorganise_subsets.py 1-194
! cat dump/train_s/deltafalse/
jsonfiles
! cat dump/train_m/deltafalse/data_unigram_5000.v2.json | python -m json.tool | less -S
utterances
utterances_small = utterances[utterances["comp"].isin([f"comp-{c}" for c in list("ijklmno")])]
utterances_small.to_csv("data/utterances_small.csv", index=False)
subsets = {split: utterances_small.loc[utterances_small["subset"] == split, "uttid"].tolist() for split in ("train", "dev", "test")}
def reorganise_subsets(jsonfiles, subsets, suffix="v2"):
    # jsonfiles = {"train": [path-to-json1, ...], "dev": ...}
    # subsets = {"train": [uttid1, ...], "dev": ...}
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
            filename = str(jsonfiles[split][dataset_id]).replace(".json", f".{suffix}.json")
            with open(filename, "w") as outfile:
                json.dump({"utts": newsplits[split]}, outfile)
            print(f"Written {filename} ({len(newsplits[split])} records)")
            
            
reorganise_subsets(jsonfiles, subsets, suffix="sm")
utterances_xsmall = utterances[utterances["comp"].isin([f"comp-{c}" for c in list("kmno")])]
utterances_xsmall.to_csv("data/utterances_xsmall.csv", index=False)
subsets = {split: utterances_xsmall.loc[utterances_xsmall["subset"] == split, "uttid"].tolist() for split in ("train", "dev", "test")}
reorganise_subsets(jsonfiles, subsets, suffix="xs")
%save -r local/RAW_reorganise_subsets.v2.py 1-208
