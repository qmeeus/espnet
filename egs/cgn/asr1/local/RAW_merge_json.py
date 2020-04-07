import json
import argparse
from pathlib import Path

"""
Given a certain number of json files, stack the outputs
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.files) >= 2
    return args


def load_json(path):
    with open(path) as jsonfile:
        return json.load(jsonfile)["utts"]

def merge_json(*paths):
    datasets = list(map(load_json, paths))
    merged = {}
    for key in datasets[0]:
        merged[key] = datasets[0][key]
        for dataset in datasets[1:]:
            merged[key]["output"].extend(dataset[key]["output"])
    return merged

def save_json(data, filename):
    with open(filename, "w") as jsonfile:
        json.dump(data, jsonfile)


if __name__ == "__main__":
    args = parse_args()
    json_files = (Path(path) for path in args.files)
    merged = merge_json(*json_files)
    save_json(merged, args.output_file)
