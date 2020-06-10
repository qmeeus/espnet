#!/usr/bin/env python3
import os
import argparse
import torchaudio
import torch
import pandas as pd
from pathlib import Path
from IPython.display import display
from multiprocessing import Pool


def existing_path(path):
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentError(path)
    return path


parser = argparse.ArgumentParser()
parser.add_argument("splits", type=existing_path, help="dataset splits")
parser.add_argument("files", type=existing_path, help="filepaths")
parser.add_argument("outdir", type=Path, help="output dir")
parser.add_argument("--njobs", type=int, default=12, help="Number of processes")
options = parser.parse_args()

utterances = (
    pd.read_csv(options.splits, index_col="uttid", usecols=[
        "uttid", "comp", "name", "start", "end", "text", "valid", "test"
    ]).join(
        pd.read_csv(options.files, names=["name", "path"], index_col="name", sep=" "),
        on="name"
    )
)

assert utterances.path.nunique() == utterances.name.nunique(), "Filenames not unique"
print(f"{utterances.path.nunique()} files and {len(utterances)} utterances")


def split_audiofile(group):
    audiofile, utterances = group
    waveform, sample_rate = torchaudio.load(audiofile)
    waveform = waveform[:1, :]

    for index, row in utterances.iterrows():
        start, end = (int(t * sample_rate) for t in (row.start, row.end))
        sample = waveform[:, start:end].clone()
        torch.save(sample, options.outdir / f"{index}.pt")


os.makedirs(options.outdir, exist_ok=True)
with Pool(processes=options.njobs) as pool:
    pool.map(split_audiofile, utterances.groupby("path"))



