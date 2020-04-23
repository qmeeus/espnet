# coding: utf-8
def split_test_by_comp(test_json):
    with open(test_json, "rb") as f:
        dataset = json.load(f)["utts"]
    components = sorted(set(map(lambda s: s["component"], dataset.values())))
    splits = {comp: {} for comp in components}
    for uttid, sample in dataset.items():
        splits[sample["component"]][uttid] = sample
    prefix = test_json.split(".")[0]
    for comp in components:
        compid = comp.split("-")[-1]
        outfile = f"{prefix}.{compid}.json"
        with open(outfile, "w") as f:
            json.dump({"utts": splits[comp]}, f)
        print(outfile, "written")
        
        
split_test_by_comp("dump/CGN_test/deltafalse/data_words.all.json")
import json
split_test_by_comp("dump/CGN_test/deltafalse/data_words.all.json")
def split_test_by_comp(test_json):
    with open(test_json, "rb") as f:
        dataset = json.load(f)["utts"]
    components = sorted(set(map(lambda s: s["component"], dataset.values())))
    splits = {comp: {} for comp in components}
    for uttid, sample in dataset.items():
        splits[sample["component"]][uttid] = sample
    prefix = test_json.split(".")[0]
    for comp in components:
        compid = comp.split("-")[-1]
        outfile = f"{prefix}.{compid}.json"
        with open(outfile, "w") as f:
            json.dump({"utts": splits[comp]}, f)
        print(f"{outfile} written ({len(splits[comp]} examples)")

        
        
def split_test_by_comp(test_json):
    with open(test_json, "rb") as f:
        dataset = json.load(f)["utts"]
    components = sorted(set(map(lambda s: s["component"], dataset.values())))
    splits = {comp: {} for comp in components}
    for uttid, sample in dataset.items():
        splits[sample["component"]][uttid] = sample
    prefix = test_json.split(".")[0]
    for comp in components:
        compid = comp.split("-")[-1]
        outfile = f"{prefix}.{compid}.json"
        with open(outfile, "w") as f:
            json.dump({"utts": splits[comp]}, f)
        print(f"{outfile} written ({len(splits[comp])} examples)")
        

        
        
split_test_by_comp("dump/CGN_test/deltafalse/data_words.all.json")
%save -r local/RAW_split_test_by_comp.py 1-8
