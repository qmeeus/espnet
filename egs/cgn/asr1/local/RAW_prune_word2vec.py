#!/usr/bin/env python

import os
import numpy as np
from time import time
from pathlib import Path

"""
Usage:
python local/RWA_prune_word2vec.py $EMBEDDING_PATH $VOCAB_FILE [--output-file $OUTPUT] [--sort]
"""

def absolute_path(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(path)
    return path


def load_vocab(vocab_file):

    def _parse(line):
        return line.split()[0]

    with open(vocab_file) as f:
        return set(map(_parse, f.readlines()))


def load_word2vec(w2v_file, vocab, special_symbols=None):
    special_symbols = special_symbols or []

    with open(w2v_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    (nvocab, dim), lines = map(int, lines[0].split()), lines[1:]

    words, vectors = [], []
    while lines:
        line = lines.pop().strip().split()
        if line[0] in vocab or line[0] in special_symbols:
            words.append(line[0])
            vectors.append(np.array(line[1:], dtype=np.float32))

    # HACK:
    if "</s>" in special_symbols and "</s>" not in words:
        words.append("</s>"); vectors.append(np.ones((dim,)))

    words, vectors = (np.array(l) for l in (words, vectors))
    assert vectors.shape == (len(vocab), dim)  # Sanity check

    print(f"{vectors.shape[0]:,} vectors loaded (dim={vectors.shape[1]})")
    vocab = sorted(vocab)
    return words.tolist(), vectors, vocab


def sort_textfile(textfile, skiplines=0):
    if skiplines > 0:
        cmd = f"""head -n {skiplines} {textfile} > {textfile}_ \\
            && tail -n +{skiplines} {textfile} | sort >> {textfile}_ \\
            && mv {textfile}_ {textfile}"""
    else:
        cmd = f"sort {textfile} > {textfile}_ && mv {textfile}_ {textfile}"

    os.system(cmd)


def prune_word2vec(w2v_file, vocab_file, sort=True, special_symbols=None, output_file="w2v_small.txt"):
    w2v_file, vocab_file = (Path(fn) for fn in (w2v_file, vocab_file))
    vocab = load_vocab(vocab_file)
    words, vectors, vocab = load_word2vec(
        w2v_file, vocab, special_symbols=list(special_symbols) if special_symbols else None
    )

    for sym, (index, value) in special_symbols.items():
        index = index if index >= 0 else len(vocab)
        if sym not in words:
            assert value is not None
            vocab.insert(index, sym)
            words.append(sym)
            vectors = np.concatenate([vectors, np.ones((1, vectors.shape[1])) * value], axis=0)
        else:
            vocab.insert(index, vocab.pop(vocab.index(sym)))

    w2v_small = Path(vocab_file.parent, output_file)
    with open(w2v_small, 'w') as w2v_file, open(vocab_file, "w") as voc_file:
        w2v_file.write("{} {}\n".format(*vectors.shape))
        for word_index, word in enumerate(vocab, 1):
            # Make sure that every index match
            voc_file.write(f"{word} {word_index}\n")
            w2v_file.write(f"{word} {' '.join(map(str, vectors[words.index(word)]))}\n")


if __name__ == '__main__':

    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("w2v", type=absolute_path, help="path to word2vec vectors")
        parser.add_argument("vocab", type=absolute_path, help="path to vocab list")
        parser.add_argument("--output-file", type=Path, default="w2v_small.txt", help="output file")
        parser.add_argument("--sort", action="store_true",
                            help="Sort output pruned vector file and vocab file.\n"
                            "Note: If you already have tokenized your texts, it WILL mess up your data!")
        return parser.parse_args()

    args = parse_args()
    w2v_file, vocab_file = args.w2v, args.vocab
    special_symbols = {"</s>": (-1, None), "<pad>": (0, 0)}
    prune_word2vec(w2v_file, vocab_file, sort=args.sort, special_symbols=special_symbols, output_file=args.output_file)
