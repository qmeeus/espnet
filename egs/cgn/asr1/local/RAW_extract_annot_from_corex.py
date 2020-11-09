# coding: utf-8
from pathlib import Path
import numpy as np
import pandas as pd
import os, sys
import argparse
import unicodedata
import string
import multiprocessing as mp
from time import time
from logger import setup
from operator import itemgetter
from copy import deepcopy


OUTPUT_DIR = Path("./data/CGN_ALL").absolute()
ANNOT_DIR = Path("~/data/cgn/CDdata/CGN_V1.0_elda_ann/data/annot/corex/sea").expanduser()
ANNOT_FILE = OUTPUT_DIR / "annotations.csv"
LOG_DIR = Path("exp/preprocessing")
LOGLEVEL = "INFO"
MIN_DURATION_SECONDS = 2
MIN_LENGTH_WORDS = 4
NJOBS = mp.cpu_count() - 1
MULTISPEAKERS_COMPONENTS = list(map("comp-{}".format, list("abcdefghi")))

logger = setup(LOG_DIR, LOGLEVEL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-existing-annotations", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--annot-dir", type=Path, default=ANNOT_DIR)
    parser.add_argument("--annot-file", type=Path, default=ANNOT_FILE)
    parser.add_argument("--min-duration-seconds", type=int, default=MIN_DURATION_SECONDS)
    parser.add_argument("--min-length-words", type=int, default=MIN_LENGTH_WORDS)
    parser.add_argument("--drop-unknown", action="store_true")
    parser.add_argument("--drop-overlapping", action="store_true")
    parser.add_argument("--deaccent", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--njobs", type=int, default=NJOBS)
    return parser.parse_args()


def load_annot(annot_file):
    with open(annot_file, encoding='latin-1') as f:
        raw_text = list(map(str.strip, f.readlines()))

    utterances = {}
    comp, lang = annot_file.split("/")[-3:-1]

    while raw_text:

        while True:

            line = raw_text.pop(0)
            if not line:
                break

            if line[0].isdigit():
                index, start, end, spkr, uttid = filter(bool, line.split(" "))
                utterances[f"{spkr}-{uttid}"] = {
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


def extract_utterances(options):
    logger.info("Gathering annotation files"); t0 = time()
    os.system(f'find {options.annot_dir} -type f -name "f*" > /tmp/seafiles.cgn')
    annot_files = pd.read_csv("/tmp/seafiles.cgn", names=["path"], squeeze=True)
    logger.info(f"{len(annot_files):,} found (took {time()-t0:.2f} s). Starting extraction"); t1 = time()
    with mp.Pool(options.njobs) as pool:
        utterances = pd.concat(
            pool.imap(load_annot, annot_files.to_list()),
            axis=0, sort=False
        ).sort_index()

    utterances.index = utterances.index.rename("uttid")
    t2 = time()
    logger.info(f"Extracted {len(utterances):,} annotations in {t2-t1:.2f} s (total={t2-t0:.2f} s).")
    os.makedirs(options.annot_file.parent, exist_ok=True)
    utterances.to_csv(options.annot_file)
    if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annots0.csv")
    return utterances


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

    connected["overlap"] = connected.apply(_is_overlapping, axis=1)
    return set(connected.loc[connected["overlap"], "uttid_x"])

def _is_different(row):
    return row["uttid_x"] != row["uttid_y"]

def _is_overlapping(row):
    return row["interval_x"].overlaps(row["interval_y"])


def filter_utterances(utterances, options):
    logger.info(f"Removing BACKGROUND and COMMENT. {len(utterances):,} utterances"); t0 = time()
    # Remove BACKGROUND and COMMENT from the utterances
    utterances = utterances[~utterances["speaker"].isin(["BACKGROUND", "COMMENT"])].copy()
    logger.info(f"Done in {time() - t0:.2f} s.")
    if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annots1.csv")

    # Extract the filename
    utterances["name"] = utterances.index.str.extract(r"-(f.\d+)\.").values

    # Reorder the columns to have annotations at the end
    column_order = [3, 4, 2, 17, 0, 1, *range(5, 17)]
    utterances = utterances[[utterances.columns[i] for i in column_order]].copy()

    def strip_accents(s):
       return ''.join(
           c for c in unicodedata.normalize('NFD', s)
           if unicodedata.category(c) != 'Mn'
       )

    def clean_text(text):
        if options.lowercase:
            text = text.lower()

        text = (
            text.replace("'m", "hem")
            .replace("'k", "ik")
            .replace("'t", "het")
            .replace("d'r", "daar")
            .replace("ggg", "xxx")
            .translate(str.maketrans('', '', string.punctuation))
        )

        if options.deaccent:
            text = strip_accents(text)

        return text

    # Clean texts: remove accents, lower, replace contractions, remove punctuation
    utterances["text"] = utterances["ORT"].map(clean_text)

    # Convert start and end to seconds and create duration and length
    utterances["start"], utterances["end"] = (utterances[col] / 1000 for col in ["start", "end"])
    utterances["duration"] = utterances["end"] - utterances["start"]
    utterances["length"] = utterances["text"].map(lambda s: len(s.split()))
    if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annots2.csv")

    def to_interval(df):
        # Tolerance is 1 second: interval limits are rounded up/down accordingly.
        def _to_interval(row):
            try:
                return pd.Interval(np.ceil(row.start), np.floor(row.end), closed='left')
            except:
                return None

        return df.apply(_to_interval, axis=1)

    if options.drop_overlapping:
        logger.info(f"Removing overlapping utterances. {len(utterances):,} utterances"); t1 = time()

        multispeakers = (
            utterances[utterances["comp"].isin(MULTISPEAKERS_COMPONENTS)]
            .reset_index(drop=False)
            .assign(interval=to_interval)
            .where(lambda df: df.interval.notnull())
            .dropna(how='all')
            [["uttid", "name", "interval"]]
            .copy()
        )

        import ipdb; ipdb.set_trace()
        # overlapping_ids = [
        #     find_overlapping_utterances(group)
        #     for group in multispeakers.groupby("name")
        # ]

        with mp.Pool(options.njobs) as pool:
            overlapping_ids = pool.map(
                find_overlapping_utterances, multispeakers.groupby("name")
            )

        overlapping_ids = set().union(*overlapping_ids)
        utterances = utterances.loc[~utterances.index.isin(overlapping_ids)].copy()
        logger.info(f"Done in {time() - t1:.2f} s.")
        if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annots3.csv")

    # Filter out utterances that contain unknown expressions (xxx)
    if options.drop_unknown:
        logger.info(f"Removing utterances with unknown words. {len(utterances):,} utterances"); t2 = time()
        utterances = utterances.loc[~utterances.text.str.contains("xxx")]
        logger.info(f"Done in {time() - t2:.2f} s.")
        if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annots4.csv")

    # Filter out utterances based on duration (seconds) and length (number of words)
    logger.info(f"Removing short utterances. {len(utterances):,} utterances"); t3 = time()
    duration_mask = utterances.duration >= options.min_duration_seconds
    length_mask = utterances.length >= options.min_length_words
    utterances = utterances[duration_mask & length_mask].copy()
    logger.info(f"Done in {time() - t3:.2f} s.")
    if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annots5.csv")

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

    logger.info(f"Removing utterances based on patterns. {len(utterances):,} utterances"); t4 = time()
    for pattern in exclude_patterns:
        utterances = utterances.loc[~utterances["text"].str.match(pattern)]
    logger.info(f"Done in {time() - t4:.2f} s.")
    if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annots6.csv")

    logger.info(f"Remaining utterances: {len(utterances):,} utterances")
    # Sort by uttid
    utterances = utterances.sort_values("uttid").copy()

    # Save and return
    utterances.to_csv(options.annot_file)
    logger.info(f"Processing completed. Elapsed time: {time() - t0:.2f} s.")
    if LOGLEVEL == "DEBUG": utterances.to_csv(LOG_DIR/"annotsF.csv")
    return utterances


if __name__ == "__main__":
    options = parse_args()
    utterances = (
        pd.read_csv(ANNOT_FILE, index_col=0)
        if options.use_existing_annotations
        else extract_utterances(options)
    )
    filter_utterances(utterances, options)

