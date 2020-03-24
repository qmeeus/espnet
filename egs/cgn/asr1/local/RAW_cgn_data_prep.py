#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cgn_root", type=Path, help="where CGN is located")
    parser.add_argument("datafiles", type=Path, help="csv file with audiofiles")
    parser.add_argument("--comp-train", type=str, help="components to include in train set")
    parser.add_argument("--comp-dev", type=str, help="components to include in dev set")
    parser.add_argument("--comp-test", type=str, help="components to include in test set")
    parser.add_argument("--lang", type=str, help="lang include in datasets")
    parser.add_argument("--tag", type=str, help="data tag (appended at the end of the created folders)")
    return parser.parse_args()


def filter_files(files, comp="a;b;c;d;e;f;g;h;i;j;k;l;m;n;o", lang="nl;vl"):
    comp_mask = files["comp"].isin(comp.split(";"))
    lang_mask = files["lang"].isin(lang.split(";"))
    selection = files[(comp_mask) & (lang_mask)]
    return selection


def create_wav_scp(files, root, output_file="wav.scp"):
    
    template = "{fileid} sox -t wav {audiofile} -b 16 -t wav - remix - |".format
    fileids = files["name"]
    audiofiles = files["audiofile"].map(lambda p: Path(root, p))

    with open(output_file, "w") as outfile:
        outfile.write("\n".join(map(template, fileids, audiofiles)))


