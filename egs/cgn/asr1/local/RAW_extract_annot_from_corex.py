# coding: utf-8
from pathlib import Path
import pandas as pd
import os, sys
import argparse
import unicodedata
import string


MIN_DURATION_SECONDS = 2
MIN_LENGTH_WORDS = 4

OUTPUT_DIR = Path("./data/CGN_ALL").absolute()
ANNOT_DIR = Path("~/data/cgn/CDdata/CGN_V1.0_elda_ann/data/annot/corex/sea").expanduser()
ANNOT_FILE = OUTPUT_DIR / "annotations.csv"

print("WARNING: Untested after refactoring. Remove this line when you are sure..."); sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--annot-dir", type=Path, default=ANNOT_DIR)
    parser.add_argument("--annot-file", type=Path, default=ANNOT_FILE)
    parser.add_argument("--min-duration-seconds", type=int, default=MIN_DURATION_SECONDS)
    parser.add_argument("--min-length-words", type=int, default=MIN_LENGTH_WORDS)
    return parser.parse_args()


def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
    

def clean_text(text):
    return strip_accents(
        text.lower()
        .replace("'m", "hem")
        .replace("'k", "ik")
        .replace("'t", "het")
        .replace("d'r", "daar")
        .replace("ggg", "xxx")
        .translate(str.maketrans('', '', string.punctuation))
    )


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
    os.system(f'find {options.annot_dir} -type f -name "fv*" > /tmp/seafiles.cgn')
    annot_files = pd.read_csv("/tmp/seafiles.cgn", names=["path"], squeeze=True)
    utterances = pd.concat(map(load_annot, annot_files), axis=0)
    utterances.index = utterances.index.rename("uttid")
    utterances.to_csv(options.annot_file)

def filter_utterances(utterances, options):

    # Remove BACKGROUND and COMMENT from the utterances
    utterances = utterances[~utterances["speaker"].isin(["BACKGROUND", "COMMENT"])].copy()

    # Extract the filename
    utterances["name"] = utterances["uttid"].str.extract(r"\w+-(fv\d+).\d+")

    # Reorder the columns to have annotations at the end
    print("WARNING: FIX COLUMN ORDER"); sys.exit(1)
    column_order = [0, 13, 16, 19, 17, 18, 14, *range(1,13)]
    utterances = utterances[[utterances.columns[i] for i in column_order]].copy()

    # Convert start and end to seconds and create duration and length
    utterances["start"], utterances["end"] = (utterances[col] / 1000 for col in ["start", "end"]) 
    utterances["duration"] = utterances["end"] - utterances["start"]
    utterances["length"] = utterances["ORT"].map(lambda s: len(s.split()))

    # Filter out utterances based on duration (seconds) and length (number of words)
    duration_mask = utterances.duration >= options.min_duration_seconds
    length_mask = utterances.length >= options.min_length_words
    utterances = utterances[duration_mask & length_mask].copy()

    # Remove some utterances based on patterns in the text
    exclude_patterns = [
        r"^([xgmhu-]+\s?)+[\.\?\!]$",               # xxx. ggg. xxx? ggg? mmh. mmh? mh? uhm. mm-hu. mmh uh. etc.
        r"^(ja\s?)+[\.\?\!]$",                      # ja. ja ja. ja? etc.
        r"^jazeker\.$",
        r"^ah[\.\?\!]$",
        r"^neen?.$",                                # nee. neen. nee? etc.
        r"^goed\.$",
        r"^allee\.$",
        r"^oké\.$",
        r"^h..?\.$",
        r"^g?oh\.$",
        r"^voil.\.$",
        r"^amai\.$",
        r"^wat.$",
        r"^oei.$",
        r"^aha.$",
        r"^pf+.$",
        r".ja.$",
        r"jawel.$",
        r"neeje.$",
        r"zeg.$",
        r"^xxx.$",
        r"^ah.$",
        r"^zo.$",
        r"^en.$",
        r"^ok..$",
        r"^ok..?$",
        r"^ai.?$",
        r"^(ja\s?)+(uh\s?)+.?$"
    ] 

    for pattern in exclude_patterns:
        utterances = utterances.loc[~utterances["ORT"].str.match(pattern)]

    utterances["text"] = utterances["ORT"].map(clean_text)

    # Sort by uttid
    utterances = utterances.sort_values("uttid").copy()
    
    # Save and return
    utterances.to_csv(options.annot_file)
    return utterances


if __name__ == "__main__":
    options = parse_args()
    extract_utterances(options)
    filter_utterances(options)
