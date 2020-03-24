import json
# coding: utf-8


def convert_json_ark_scp(jsonpath):

    def update_input(key, example):
        feat = example["input"][0]["feat"]
        feat = feat.split(":")[0].replace(".ark", ".scp") + ":" + key
        example["input"][0]["feat"] = feat
        example["input"][0]["filetype"] = "scp"
        return example

    with open(jsonpath) as infile:
        metadata = json.load(infile)
        keys, samples = zip(*metadata["utts"].items())
        metadata["utts"] = dict(zip(keys, map(update_input, keys, samples)))

    newfile = jsonpath.replace(".json", ".scp.json")

    with open(newfile, "w") as outfile:
        json.dump(metadata, outfile)

    return newfile


get_datafile = "dump/{subset}/deltafalse/{filename}".format

for subset in ("train_s", "dev_s", "train_m", "dev_m"):
    for name in ("data", "data_unigram_5000"):
        jsonpath = get_datafile(subset=subset, filename=name + ".json")
        try:
            outfile = convert_json_ark_scp(jsonpath)
            print(f"saved {outfile}")
        except Exception as err:
            print(f"Error with {jsonpath}: {err}")


# %save -r local/RAW_convert_json_ark_scp.py 1-25
