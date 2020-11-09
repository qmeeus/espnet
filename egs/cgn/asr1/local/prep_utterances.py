import argparse
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import string
from time import time
import unicodedata

from logger import setup


DEFAULTS = argparse.Namespace(**{
    "logdir": Path("exp/preprocessing"),
    "loglevel": "INFO",
    "annotation_dir": Path("~/data/cgn/CDdata/CGN_V1.0_elda_ann/data/annot/corex/sea").expanduser(),
    "annotation_file": Path("data/CGN_ALL/annotations.csv"),
    "njobs": mp.cpu_count() - 1,
    "lowercase": True,
    "remove_punctuation": True,
    "deaccent": True,
    "min_duration_seconds": 2,
    "min_length_words": 4,
})


logger = setup(DEFAULTS.logdir, DEFAULTS.loglevel)


def log_stats(count=True):
    def wrapper(func):
        def _wrapper(*args, **kwargs):
            logger.info(f"Task {func.__name__} is running...")
            t0 = time()
            out = func(*args, **kwargs)
            msg = f"Task {func.__name__} took {time() - t0:.2f} s. "
            if count:
                msg += f"{len(out):,} utterances."
            logger.info(msg)
            return out
        return _wrapper
    return wrapper


@log_stats()
def extract_utterances(options):

    utterances = load_utterances(options)

    mandatory_steps = [
        remove_background_and_comments,
        extract_text,
        add_time_info
    ]

    optional_steps = [
        remove_overlapping_utterances,
        remove_unknown,
        remove_short_utterances,
    ]

    preprocessing_steps = mandatory_steps + optional_steps

    for step in preprocessing_steps:
        utterances = step(utterances, options)

    columns = [
        'comp', 'lang', 'speaker', 'name', 'text', 'duration', 'length',
        'start', 'end', 'ORT', 'WID', 'POS', 'LEM', 'LID', 'NID', 'MAR',
        'BEG', 'END', 'PHO', 'PR1', 'PR2'
    ]

    utterances = utterances[columns].copy()
    to_csv(utterances, options.annotation_file)
    return utterances


@log_stats():
def load_utterances(options):
    os.system(f'find {options.annotation_dir} -type f -name "f*" > /tmp/seafiles.cgn')
    annot_files = pd.read_csv("/tmp/seafiles.cgn", names=["path"], squeeze=True)

    with mp.Pool(options.njobs) as pool:
        utterances = pd.concat(
            pool.imap(load_annotations, annot_files.to_list()),
            axis=0, sort=False
        ).sort_index()

    return utterances

def load_annotations(annotation_file):

    with open(annotation_file, encoding='latin-1') as f:
        raw_text = list(map(str.strip, f.readlines()))

    utterances = {}
    comp, lang = annotation_file.split("/")[-3:-1]

    while raw_text:

        while True:

            line = raw_text.pop(0)
            if not line:
                break

            if line[0].isdigit():
                index, start, end, spkr, uttid = filter(bool, line.split(" "))
                utterances[f"{spkr}-{uttid}"] = {
                    'name': uttid.split(".")[0],
                    'start': int(start),
                    'end': int(end),
                    'speaker': spkr,
                    'comp': comp,
                    'lang': lang
                }

            else:
                try:
                    key, value = line.split(" ", maxsplit=1)
                    utterances[f"{spkr}-{uttid}"][key] = value
                except:
                    pass

    return pd.DataFrame.from_dict(utterances, orient="index")


@log_stats()
def remove_background_and_comments(utterances, options):
    return utterances[~utterances["speaker"].isin(["BACKGROUND", "COMMENT"])].copy()


@log_stats()
def remove_unknown(utterances, options):
    return utterances.loc[~utterances.text.str.contains("xxx")].copy()


@log_stats()
def remove_short_utterances(utterances, options):
    duration_mask = utterances.duration >= options.min_duration_seconds
    length_mask = utterances.length >= options.min_length_words
    return utterances[duration_mask & length_mask].copy()


@log_stats()
def filter_patterns(utterances, options):

    # Remove some utterances based on patterns in the text
    exclude_patterns = [
        r"^([xgmhu-]+\s?)+$",       # xxx. ggg. xxx? ggg? mmh. mmh? mh? uhm. mm-hu. mmh uh. etc.
        r"^(ja\s?)+$",              # ja. ja ja. ja? etc.
        r"^jazeker$",
        r"^ah$",
        r"^neen?$",                 # nee. neen. nee? etc.
        r"^goed$",
        r"^allee$",
        r"^oké$",
        r"^h..?$",
        r"^g?oh$",
        r"^voil.$",
        r"^amai$",
        r"^wat$",
        r"^oei$",
        r"^aha$",
        r"^pf+$",
        r".ja$",
        r"jawel$",
        r"neeje$",
        r"zeg$",
        r"^xxx$",
        r"^ah$",
        r"^zo$",
        r"^en$",
        r"^ok\S*$",
        r"^ai.?$",
        r"^(ja\s?)+(uh\s?)+$"
    ]

    for pattern in exclude_patterns:
        utterances = utterances.loc[~utterances["text"].str.match(pattern)]

    return utterances.copy()


@log_stats(count=False)
def add_time_info(utterances, options):
    utterances.index = utterances.index.rename("uttid")
    utterances["start"], utterances["end"] = (utterances[col] / 1000 for col in ["start", "end"])
    utterances["duration"] = utterances["end"] - utterances["start"]
    utterances["length"] = utterances["text"].map(lambda s: len(s.split()))
    return utterances


@log_stats(count=False)
def extract_text(utterances, options):

    def deaccent(text):
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def clean_text(text):
        if options.lowercase:
            text = text.lower()

        replace_dict = {
            "'m": "hem",
            "'k": "ik",
            "'t": "het",
            "d'r": "daar",
            "ggg": "xxx"

        }

        for key, value in replace_dict.items():
            text = text.replace(key, value)

        if options.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        if options.deaccent:
            text = deaccent(text)

        return text

    utterances["text"] = utterances["ORT"].map(clean_text)
    return utterances


@log_stats()
def remove_overlapping_utterances(utterances, options):

    def to_interval(df):
        # Tolerance is 1 second: interval limits are rounded up/down accordingly.
        def _to_interval(row):
            try:
                return pd.Interval(np.ceil(row.start), np.floor(row.end), closed='left')
            except:
                return None

        return df.apply(_to_interval, axis=1)

    multispeakers = (
        utterances.reset_index(drop=False)
        .assign(interval=to_interval)
        .where(lambda df: df.interval.notnull())
        .dropna(how='all')
        [["uttid", "name", "interval"]]
        .copy()
    )

    with mp.Pool(options.njobs) as pool:
        overlapping_ids = pool.map(
            find_overlapping_utterances, multispeakers.groupby("name")
        )

    overlapping_ids = set().union(*overlapping_ids)
    return utterances.loc[~utterances.index.isin(overlapping_ids)].copy()


def find_overlapping_utterances(group):
    """
    Input: tuple (index, group) such as given by pd.DataFrame.groupby
    Returns: a set of ids that overlap
    """
    if isinstance(group, tuple):
        _, group = group

    connected = (
        group.merge(group, on="name")
        .where(_is_different)
        .dropna(how="all")
    )

    try:
        connected["overlap"] = connected.apply(_is_overlapping, axis=1)
        return set(connected.loc[connected["overlap"], "uttid_x"])
    except:
        return set()


def _is_different(row):
    return row["uttid_x"] != row["uttid_y"]

def _is_overlapping(row):
    return row["interval_x"].overlaps(row["interval_y"])

def to_csv(dataframe, filename):
    filename = Path(filename)
    os.makedirs(filename.parent, exist_ok=True)
    dataframe.sort_index().to_csv(filename)


if __name__ == "__main__":
    extract_utterances(DEFAULTS)
