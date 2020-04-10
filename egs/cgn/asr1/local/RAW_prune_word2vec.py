import os
import numpy as np
from time import time
from pathlib import Path


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

    with open(w2v_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    (nvocab, dim), lines = map(int, lines[0].split()), lines[1:]

    words, vectors = [], []
    while lines:
        line = lines.pop().strip().split()
        if line[0] in vocab or line[0] in special_symbols:
            words.append(line[0])
            vectors.append(np.array(line[1:], dtype=np.float32))

    words, vectors = (np.array(l) for l in (words, vectors))
    assert vectors.shape == (len(vocab), dim)  # Sanity check
    return words, vectors


def sort_textfile(textfile):
    command = f"sort {textfile} > {textfile}_ && mv {textfile}_ {textfile}"
    os.system(command)


def prune_word2vec(w2v_file, vocab_file, sort=True, eos="</s>", unk="<unk>"):
    w2v_file, vocab_file = (Path(fn) for fn in (w2v_file, vocab_file))
    vocab = load_vocab(vocab_file)
    words, vectors = load_word2vec(w2v_file, vocab, special_symbols=[eos, unk])

    new_name = Path(vocab_file.parent, "w2v_small.txt")
    with open(new_name, 'w') as f:
        f.write("{} {}\n".format(*vectors.shape))
        for i in range(len(vectors)):
            f.write(f"{words[i]} {' '.join(map(str, vectors[i]))}\n")

    if sort:
        sort_textfile(new_name)
        sort_textfile(vocab_file)


if __name__ == '__main__':

    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("w2v", type=absolute_path, help="path to word2vec vectors")
        parser.add_argument("vocab", type=absolute_path, help="path to vocab list")
        parser.add_argument("--sort", action="store_true", 
                            help="Sort output pruned vector file and vocab file.\n" 
                            "Note: If you already have tokenized your texts, it WILL mess up your data!")
        return parser.parse_args()

    args = parse_args()
    w2v_file, vocab_file = args.w2v, args.vocab
    prune_word2vec(w2v_file, vocab_file, sort=args.sort)
