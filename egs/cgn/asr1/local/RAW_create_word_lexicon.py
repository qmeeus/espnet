import os
import json
from collections import Counter, defaultdict
from tqdm import tqdm
from pathlib import Path
import argparse
from functools import partial
from unidecode import unidecode


"""
Usage:
python local/RAW_create_word_lexicon.py --source-dataset data_unigram_5000.\{\}.json --subsets train valid
python local/RAW_create_word_lexicon.py --source-dataset data_unigram_5000.\{\}.json --subsets test --datatags a b f g h i j k l m n o --use-existing-vocab
"""


DROP_REASON = {
    "blacklist": 0,
    "unknown": 0,
    "length": 0
}

UNKNOWN_WORDS = set()
MIN_OUTPUT_LENGTH = 1

def absolute_path(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(path)
    return path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab-file", type=Path,
                        default="data/lang_word/CGN_train_word_units.txt",
                        help="Where to save the vocabulary file")

    parser.add_argument("--use-existing-vocab", action="store_true")
    parser.add_argument("--embeddings", type=absolute_path,
                        default="~/spchdisk/data/dutchembeddings/CoNLL17/model.txt",
                        help="Path to word vectors.")

    parser.add_argument("--known-words", type=absolute_path,
                        default="~/spchdisk/data/dutchembeddings/CoNLL17/vocab.txt",
                        help="Path to embeddings vocab. This file can be created in bash with:\n"
                        "`tail -n+1 $embedding_file | cut -d ' ' -f1 > $vocab`")

    parser.add_argument("--eos", default="</s>", type=str, help="EOS symbol")
    parser.add_argument("--eos-index", default=-1, type=int, help="EOS index (-1 for last)")
    parser.add_argument("--eos-value", default=None, type=float, help="EOS value (constant)")
    parser.add_argument("--pad", default="<pad>", type=str, help="PAD symbol")
    parser.add_argument("--pad-index", default=0, type=int, help="PAD index (-1 for last)")
    parser.add_argument("--pad-value", default=0, type=float, help="PAD value (constant)")
    parser.add_argument("--unk", default="<unk>", type=str, help="UNK symbol")
    parser.add_argument("--unk-index", default=0, type=int, help="UNK index (-1 for last)")
    parser.add_argument("--unk-value", default=None, type=float, help="UNK value (constant)")

    parser.add_argument("--datatags", nargs="*", default=["all", "mono", "ok", "o"],
                        help="The tag of the dataset, used to identify the json file")

    parser.add_argument("--source-dataset", type=str,
                        default="data_unigram_1000.{}.json",
                        help="What json file to use in the data directory. "
                        "{} is replaced by one of DATATAGS and should be escaped in bash.")

    parser.add_argument("--json-prefix", type=str, default="data_words")

    parser.add_argument("--subsets", nargs="*",
                        default=["train", "valid", "test"],
                        help="Which subsets to process")

    parser.add_argument("--dataset-path", type=str,
                        default="dump/CGN_{}/deltafalse",
                        help="How to find the right path to the dataset dump. "
                        "{} is replaced by one of SUBSETS and should be escaped in bash.")

    parser.add_argument("--remove-dash", action="store_true",
                        help="Replace dashes (-) with spaces in sentences before parsing")

    parser.add_argument("--deaccent", action="store_true",
                        help="Remove accent from sentences")

    parser.add_argument("--unknown-action", default="drop",
                        choices=["drop", "ignore"],
                        help="What to do with the words not present in the embeddings file.\n"
                        "drop: the sentence is not added to the dataset\n"
                        "ignore: the word is replaced with the special symbol <unk>")

    parser.add_argument("--blacklist", nargs="*",
                        default=["xxx", "ggg"],
                        help="Special words that if present will cause to drop the sentence.")

    parser.add_argument("--ignored-words", nargs="*",
                        default=["uh", "mmh", "uhm"],
                        help="Special tokens that will be filtered out of the parsed sentence.")

    return parser.parse_args()


def load_known_words(vocab_file):
    with open(vocab_file, encoding='latin-1') as infile:
        return set(filter(bool, map(str.strip, infile.readlines())))


def split_sentence(sentence, remove_dash=False, deaccent=False, ignored_words=None):
    if deaccent:
        sentence = unidecode(sentence)

    if remove_dash:
        sentence = sentence.replace("-",  " ")

    words = sentence.split()

    if ignored_words:
        words = list(filter(lambda word: word not in ignored_words, words))

    return words


def parse_sentence(sentence, word2token, tokenize=str.split, blacklist=None, unknown_action="drop", unk="<unk>"):

    words = tokenize(sentence)

    if blacklist and any(word in blacklist for word in words):
        DROP_REASON["blacklist"] += 1
        return

    unknown_words = [w for w in words if w not in word2token]
    if unknown_action == "drop" and unknown_words:
        DROP_REASON["unknown"] += 1
        for w in unknown_words:
            UNKNOWN_WORDS.add(w)
        return
    elif unknown_action == "ignore":
        assert unk is not None

    # HACK: unk_index will be None if not present but we don't care because the function should never arrive
    # there if there are unknown words. The unk is not in word2token only if unknown_action is drop
    unk_index = word2token.get(unk)
    tokens = list(map(lambda w: word2token[w] if w in word2token else unk_index, words))
    # assert all(token is not None for token in tokens)  # Sanity check, uncomment if you're not sure

    if len(tokens) < MIN_OUTPUT_LENGTH:
        DROP_REASON["length"] += 1
        return

    return {
        "shape": [len(tokens), len(word2token)],
        "text": sentence,
        "token": " ".join(words),
        "tokenid": " ".join(map(str, tokens))
    }


def load_vocab(vocab_file):
    with open(vocab_file) as f:
        return dict(map(str.split, map(str.strip, f.readlines())))


def build_vocab(sentences, known_words, output_file, tokenize=str.split,
                eos="</s>", unk="<unk>", eos_index=-1, unk_index=0):

    counter = Counter()
    for sent in sentences:
        words = tokenize(sent)
        counter.update(filter(lambda word: word in known_words, words))

    lexicon = sorted(counter)

    for sym, idx in [(eos, eos_index), (unk, unk_index)]:

        if sym:
            lexicon.insert(idx if idx != -1 else len(lexicon), (
                lexicon.pop(lexicon.index(sym))
                if sym in lexicon else sym
            ))

    with open(output_file, 'w') as f:
        for i, word in enumerate(lexicon, 1):
            f.write(f"{word} {i}\n")

    print(f"Lexicon saved to {output_file} ({len(lexicon):,} words)")
    return output_file

def read_metadata(input_file:Path):
    with open(input_file) as infile:
        return json.load(infile)


def iter_sentences(*filenames):
    for filename in filenames:
        for sample in read_metadata(filename)["utts"].values():
            yield sample["output"][0]["text"]


if __name__ == '__main__':

    options = parse_args()
    if options.unknown_action == "drop":
        # HACK: see parse_sentence's unk_index
        options.unk = None

    tokenize = partial(
        split_sentence,
        remove_dash=options.remove_dash,
        deaccent=options.deaccent,
        ignored_words=options.ignored_words
    )

    known_words = load_known_words(options.known_words)
    print(f"{len(known_words):,} known words loaded")

    sentences = iter_sentences(*[
        Path(options.dataset_path.format(subset), options.source_dataset.format(tag))
        for subset in options.subsets
        for tag in options.datatags
    ])

    if not options.use_existing_vocab:
        build_vocab(
            sentences,
            known_words,
            options.vocab_file,
            tokenize=tokenize,
            eos=options.eos,
            unk=options.unk,
            eos_index=options.eos_index,
            unk_index=options.unk_index
        )

        from RAW_prune_word2vec import prune_word2vec
        special_symbols = {
            options.eos: (options.eos_index, options.eos_value),
            options.pad: (options.pad_index, options.pad_value)
        }
        prune_word2vec(options.embeddings, options.vocab_file, sort=True, special_symbols=special_symbols)

    word2token = load_vocab(options.vocab_file)

    for tag in options.datatags:
        input_json = options.source_dataset.format(tag)

        for subset in options.subsets:

            total = ignored = 0
            data_dir = options.dataset_path.format(subset)
            metadata = read_metadata(Path(data_dir, input_json))["utts"]
            new_dataset = {}
            ids, samples = map(list, zip(
                *sorted(metadata.items(), key=lambda t: t[0])
            ))

            sentences = (sample["output"][0]["text"] for sample in samples)
            progress_bar = tqdm(zip(ids, sentences), total=len(ids), desc=f"{subset}.{tag}")
            for uttid, sentence in progress_bar:

                parsed = parse_sentence(
                    sentence, word2token,
                    tokenize=tokenize,
                    blacklist=options.blacklist,
                    unknown_action=options.unknown_action,
                    unk=options.unk
                )

                total += 1
                if not parsed:
                    ignored += 1
                    continue

                new_dataset[uttid] = metadata[uttid]
                new_dataset[uttid]["output"][0] = dict(
                    name="target1",
                    **parsed
                )

            print(f"{ignored:,}/{total:,} sentences were dropped ({ignored / total:.2%})")

            newfile = f"{data_dir}/{options.json_prefix}.{tag}.json"
            with open(newfile, "w") as outfile:
                json.dump({"utts": new_dataset}, outfile)

            print(f"{len(new_dataset)} utterances saved to {newfile}")

    unknown_words_file = Path(options.vocab_file.parent, "unknown_words.txt")
    with open(unknown_words_file, "w") as f:
        f.write("\n".join(sorted(UNKNOWN_WORDS)))

    print("{blacklist:,} blacklisted, {unknown:,} unknown, {length:,} too short".format(**DROP_REASON))
    print(f"{len(UNKNOWN_WORDS):,} unknown words saved as {unknown_words_file}")
