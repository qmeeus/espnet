# coding: utf-8
import os
import json
import pandas as pd
import argparse
from pathlib import Path


# TODO: Change to: one dataset per component and concatenate datasets when loading
# TODO: This allows to have more flexible options (eg use gradually more components, apply CMVN per component, etc.)
# TODO: Move to torch.utils.data will simplify this (it implements ConcatDataset) --> see my developments in transformers
# TODO: Target directory structure:
# TODO: cgn/comp-{a-o}/{train,valid,test}/{feats.ark,feats.scp,text}
# TODO: tokenization is made at loading using huggingface tokenizers (instead of char_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_prefix", type=str, help="prefix of global json file")
    parser.add_argument("--splits", type=Path, default="data/CGN_ALL/splits.csv", help="CSV file with splits")
    parser.add_argument("--groups", nargs="*", default=["o", "ok", "jklmno"], help="components to include in each dataset")
    parser.add_argument("--group-names", nargs="*", default=["o", "ok", "mono"], help="dataset tags")
    parser.add_argument("--dump", type=Path, default="dump", help="location of dump directory")
    parser.add_argument("--subsets", nargs="*", default=["train", "valid"], help="subsets on which to apply the split")
    args = parser.parse_args()
    assert len(args.groups) == len(args.group_names)
    return args

def main():
    options = parse_args()
    splits = pd.read_csv(options.splits, index_col="uttid")
    for subset in options.subsets:
        # subset_dir = Path(f"dump/CGN_{subset}/deltafalse")
        subset_dir = Path(f"dump/CGN_{subset}/nopitch")
        global_json = subset_dir / f"{options.json_prefix}.json"
        with open(global_json, encoding='utf-8') as f:
            dataset = json.load(f)["utts"]

        for group_name, group in zip(options.group_names, options.groups):
            uttids = set(splits.loc[splits.comp.isin([f"comp-{x}" for x in list(group)])].index)
            selection = {uttid: utt for uttid, utt in dataset.items() if uttid in uttids}
            filename = Path(subset_dir, f"{options.json_prefix}.{group_name}.json")
            with open(filename, "w") as outfile:
                json.dump({"utts": selection}, outfile, indent=4, sort_keys=True)

            print(f"{len(selection)} items saved to {filename}")

        os.rename(global_json, str(global_json).replace(".json", ".all.json"))

if __name__ == "__main__":
    main()
