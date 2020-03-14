# coding: utf-8
import json
import gzip
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_annotation_files(tag_dir):
    tag_files = list(tag_dir.glob("comp-k/vl/*.tag.gz"))
    tag_files.extend(list(tag_dir.glob("comp-o/vl/*.tag.gz")))
    tag_files = dict(map(lambda v: (v.stem.replace(".tag", ""), v), tag_files))
    return tag_files

def parse_json(filepath):
    with open(filepath) as json_file:
        metadata = json.load(json_file)["utts"]
    keys, metadata = map(
        list, zip(*sorted(metadata.items(), key=lambda t: t[0]))
    )

    return keys, metadata

def key2file(key):
    return key2uttid(key).split(".")[0]

def key2uttid(key):
    return key.split("-")[1]

def build_dict(freq_file, insert_unk=True, save_to=None):
    with open(freq_file) as f:
        tags = list(map(
            lambda s: s.split(maxsplit=1)[1],
            filter(bool, map(str.strip, f.readlines()))
        ))

    if insert_unk:
        tags.insert(0, "<unk>")

    if save_to:
        with open(save_to, "w") as f:
            for i, tag in enumerate(tags, 1):
                f.write(f"{tag} {i}\n")

    return tags


def make_target(jsonfile, tag_files, target_dict):
    keys, metadata = parse_json(jsonfile)

    current_file = None
    for i, key in enumerate(tqdm(keys)):
        stem = key2file(key)
        if stem != current_file:
            with gzip.open(tag_files[stem], "rb") as f:
                soup = BeautifulSoup(f.read().decode('latin-1'), 'lxml')
                current_file = stem

        utt = soup.find("pau", {"ref": key2uttid(key)})
        pos_tags = list(map(lambda t: t.get("pos"), utt.find_all("pw")))
        tokenids = [target_dict.index(tag) for tag in pos_tags]
        metadata[i]["output"][0]["token"] = " ".join(pos_tags)
        metadata[i]["output"][0]["tokenid"] = " ".join(map(str, tokenids))
        metadata[i]["output"][0]["shape"] = [len(tokenids), len(target_dict)]

    with open(jsonfile.parent.joinpath("pos_tags.json"), "w") as f:
        json.dump({"utts": dict(zip(keys, metadata))}, f)

def main():

    project_root = Path("~/spchdisk/repos/espnet/egs/cgn/asr1").expanduser()
    data_root = Path("~/data/cgn/CDdata/CGN_V1.0_elda_ann/data").expanduser()
    tag_frequencies = Path(data_root, "lexicon/freqlists/tagalph.frq")
    tag_dir = data_root.joinpath("annot/xml/tag/")
    json_train, json_valid = (
        Path(project_root, f"dump/{subset}/deltafalse/data.json")
        for subset in ("train_s_", "dev_s")
    )

    target_dict = build_dict(tag_frequencies, save_to=project_root.joinpath("data/pos_tags.txt"))
    tag_files = get_annotation_files(tag_dir)
    for jsonfile in (json_train, json_valid):
        make_target(jsonfile, tag_files, target_dict)


if __name__ == '__main__':
    main()

