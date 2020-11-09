from pathlib import Path
import pandas as pd
import gzip
from bs4 import BeautifulSoup
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import os
import argparse


NJOBS = mp.cpu_count()
CGN = Path("~/data/cgn").expanduser().resolve()
CSV_DUMP = Path("/users/spraak/qmeeus/spchdisk/repos/espnet/egs/cgn/asr1/data/datafiles.csv")
ANNOT_DIR = Path(CGN, "CDdata/CGN_V1.0_elda_ann/data")
SPKR_META = Path(ANNOT_DIR, "meta/text/speakers.txt")
ANNOT_XML = Path(ANNOT_DIR, "annot/xml")
ORT, TAG = (Path(ANNOT_XML, d) for d in ("skp-ort", "tag"))
OUTPUT_DIR = Path("/users/spraak/qmeeus/spchdisk/repos/espnet/egs/cgn/asr1/data/CGN_ALL")
ANNOT_FILE = Path(OUTPUT_DIR, "annotations.csv")

FILTER_LANG = ["vl", "nl"]
FILTER_COMP = []
MIN_LENGTH_SEC = 2
MIN_WORDS = 2
MAX_FILES = 0
DROP_UNKNOWN_SPEAKERS = True


def path_type(should_exist=False, resolve=False):
    def is_valid(p):
        p = Path(p)
        p = p.expanduser().resolve() if resolve else p
        if should_exist and not p.exists():
            raise FileNotFoundError(p)
        return p
    return is_valid


def load_files_meta(filelist):
    files = pd.read_csv(filelist)
    del files["text"]

    if FILTER_LANG:
        files = files[files["lang"].isin(FILTER_LANG)]

    if FILTER_COMP:
        files = files[files["comp"].isin(FILTER_COMP)]

    if MAX_FILES:
        files = files.head(MAX_FILES)

    return files


def load_speakers():
    return pd.read_csv(SPKR_META, sep="\t")


def extract_annot(df):

    raise ValueError("Deprecated. Use RAW_extract_annot_from_corex.py")

    def generate_annotations(row):
        comp, lang, name = (row[key] for key in ("comp", "lang", "name"))
        partial_path = Path(comp, lang)
        ort = Path(ORT, partial_path, name + ".skp.gz")
        tag = Path(TAG, partial_path, name + ".tag.gz")

        if not all(f.exists() for f in (ort, tag)):
            return

        with gzip.open(ort) as ortfile, gzip.open(tag) as tagfile:
            ort_soup = BeautifulSoup(ortfile.read(), 'lxml')
            tag_soup = BeautifulSoup(tagfile.read(), 'lxml')
            for utt in ort_soup.find_all("tau"):
                uttid = utt.get("ref")
                spk = utt.get("s")
                start = utt.get("tb")
                end = utt.get("te")

                if MIN_LENGTH_SEC > 0 and float(end) - float(start) < MIN_LENGTH_SEC:
                    continue

                words = [w.get("w").lower() for w in utt.find_all("tw")]
                pos = [w.get("pos") for w in tag_soup.find("pau", {"ref": uttid}).find_all("pw")]

                if MIN_WORDS > 0 and len(words) < MIN_WORDS:
                    continue

                yield {
                    "comp": comp,
                    "lang": lang,
                    "name": name,
                    "uttid": f"{spk}-{uttid}",
                    "speaker": spk,
                    "start": start,
                    "end": end,
                    "text": " ".join(words),
                    "pos": " ".join(pos)
                }

    def _extract(row):
        return list(generate_annotations(row))

    annotations = pd.DataFrame(sum(df.apply(_extract, axis=1).to_list(), []))
    return annotations.set_index("uttid")


def to_str(df):
    return df.apply(lambda row: " ".join([str(row[col]) for col in df.columns]), axis=1)


def create_wav_scp(files):

    def _format(wavfile):
        wavfile = Path(wavfile)
        return f"{wavfile.stem} sox -t wav {Path(CGN, wavfile)} -b 16 -t wav - remix - |\n"

    with open(Path(OUTPUT_DIR, "wav.scp"), 'w') as f:
        for line in files.sort_values("name")["audio"].map(_format):
            f.write(line)


def create_annotations(files):

    with mp.Pool(processes=NJOBS) as pool:
        splits = np.array_split(files, NJOBS)
        pool_results = list(tqdm(pool.imap(extract_annot, splits), total=NJOBS))

    annotations = (
        pd.concat(pool_results, axis=0).reset_index()
        .assign(_uttid=lambda df: df["uttid"].map(lambda s: int(s.split(".")[-1])))
        .sort_values(["speaker", "name", "_uttid"])
        .drop("_uttid", axis=1)
    )

    if DROP_UNKNOWN_SPEAKERS:
        annotations = annotations.where(annotations["speaker"] != "UNKNOWN").dropna(how="all")

    annotations.to_csv(ANNOT_FILE, index=False)
    return annotations


def write(data:pd.DataFrame, filename:str):
    row_to_string = lambda row: " ".join([str(row[col]) for col in data.columns]) + "\n"
    filename = Path(OUTPUT_DIR, filename)
    with open(filename, "w") as f:
        for line in data.apply(row_to_string, axis=1):
            f.write(line)
    print(f"{len(data)} lines written to {filename}")


def create_all_files(filelist):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = load_files_meta()
    create_wav_scp(files)

    annotations = create_annotations(files)
    create_utt2spk(annotations)
    create_spk2utt(annotations)
    create_segments(annotations)
    create_text(annotations)


def create_utt2spk(annotations):
    utt2spk = annotations[["uttid", "speaker"]]
    write(utt2spk, "utt2spk")
    return utt2spk


def create_spk2utt(annotations):
    spk2utt = (
        annotations.groupby("speaker")
        .agg({"uttid": lambda g: " ".join(g)})
        .reset_index()
        .sort_values("speaker")
    )
    write(spk2utt, "spk2utt")
    return spk2utt

def create_utt2dur(annotations):
    utt2dur = annotations[["uttid", "duration"]].copy()
    utt2dur["duration"] = utt2dur["duration"].map("{:.3f}".format)
    write(utt2dur, "utt2dur")


def create_segments(annotations):
    segments = annotations[["uttid", "name", "start", "end"]].copy()
    segments.update(segments[["start", "end"]].applymap("{:.3f}".format))
    write(segments, "segments")
    return segments


def create_text(annotations):
    texts = annotations[["uttid", "text"]]
    write(texts, "text")
    return texts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot-file", default=ANNOT_FILE, type=Path)
    parser.add_argument("--file-list", default=CSV_DUMP, type=Path)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--components", nargs="*", default=list("abefghijklmno"), type=list)
    parser.add_argument("--lang", nargs="*", default=["vl"], type=list)
    parser.add_argument("--use-existing-annot", action="store_true")
    parser.add_argument("--use-existing-file-registry", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':

    options = parse_args()

    os.makedirs(options.output_dir, exist_ok=True)

    files = (pd.read_csv(options.file_list)
             if options.use_existing_file_registry
             else load_files_meta())

    annotations = (
        (pd.read_csv(options.annot_file)
         if options.use_existing_annot
         else create_annotations(files))
        .where(lambda df: df["comp"].isin(list(map("comp-{}".format, options.components))))
        .dropna(how='all').sort_values("uttid")
    )

    files = (
        files.loc[files["name"].isin(annotations["name"].unique())]
        .sort_values("name").copy()
    )

    create_wav_scp(files)

    for func in (create_utt2spk, create_spk2utt, create_utt2dur, create_segments, create_text):
        func(annotations)

    backup_dir = options.output_dir / ".backup"
    os.makedirs(backup_dir, exist_ok=True)
    for filename in ["utt2dur", "utt2num_frames", "feats.scp"]:
        fullpath = options.output_dir / filename
        if fullpath.exists():
            print(f"Moving {fullpath} to {backup_dir}")
            os.rename(fullpath, backup_dir / filename)

