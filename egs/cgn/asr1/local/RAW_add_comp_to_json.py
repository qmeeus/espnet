# coding: utf-8
import pandas as pd
utterances = pd.read_csv("data/utterances.csv")
utterances
def load_dataset(jsonpath):
    with open(jsonpath, "rb") as f:
        return json.load(f)["utts"]
        
utterances.set_index("uttid", inplace=True)
def add_component(dataset):
    for k in dataset:
        dataset[k]["component"] = utterances.loc[k, "comp"]
    
f_json = "dump/CGN_{subset}/deltafalse/data_{target}.{tag}.json".format
f_json = "dump/CGN_{0}/deltafalse/data_{1}.{2}.json".format
ok_train = f_json("train", "words", "o")
ok_train = load_dataset(f_json("train", "words", "o"))
import json
ok_train = load_dataset(f_json("train", "words", "ok"))
len(ok_train)
add_component(ok_train)
list(filter(lambda uttid: uttid not in utterances.index, list(ok_train)))
utterances
utterances[utterances.comp in ("comp-o", "comp-k")]
utterances[utterances.comp.isin(["comp-o", "comp-k"])]
errors = list(filter(lambda uttid: uttid not in utterances.index, list(ok_train)))
len(errors)
errors[:10]
utterances[utterances.name == "fv600270"]
ls data/CGN_ALL/
!head data/CGN_ALL/annotations.csv
utterances = pd.read_csv("data/CGN_ALL/annotations.csv", index_col="uttid")
utterances
errors = list(filter(lambda uttid: uttid not in utterances.index, list(ok_train)))
len(errors)
add_component(ok_train)
ok_train["V60036-fv601218.10"]
def main():
    utterances = pd.read_csv("data/CGN_ALL/annotations.csv", index_col="uttid") 
    for subset in ("train", "valid", "test"):
        for tag in ("o", "ok", "mono", "all"):
            jsonpath = f_json(subset, "words", tag)
            dataset = load_dataset(jsonpath)
            add_component(dataset)
            with open(jsonpath, "w") as f:
                json.dump(dataset, f, indent=4, sort_keys=True)
            print(f"{jsonpath} written")
            
            
main()
ok_train = load_dataset(f_json("train", "words", "ok"))
def load_dataset(jsonpath):
    with open(jsonpath, "rb") as f:
        return json.load(f)
        
        
ok_train = load_dataset(f_json("train", "words", "ok"))
def main():
    utterances = pd.read_csv("data/CGN_ALL/annotations.csv", index_col="uttid") 
    for subset in ("train", "valid", "test"):
        for tag in ("o", "ok", "mono", "all"):
            jsonpath = f_json(subset, "words", tag)
            dataset = load_dataset(jsonpath)
            add_component(dataset)
            with open(jsonpath, "w") as f:
                json.dump({"utts": dataset}, f, indent=4, sort_keys=True)
            print(f"{jsonpath} written")
                        
main()
def load_dataset(jsonpath):
    with open(jsonpath, "rb") as f:
        return json.load(f)["utts"]
        
        
        
ok_train = load_dataset(f_json("train", "words", "ok"))
ok_train
%save -r local/RAW_add_comp_to_json.py 1-41
