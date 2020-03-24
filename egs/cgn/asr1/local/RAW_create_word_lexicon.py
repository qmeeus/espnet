# coding: utf-8
import os
import json
from collections import Counter
from collections import defaultdict
from tqdm import tqdm


def create_vocab(sentences, output_dir, subset):
    counter = Counter()
    for sent in sentences:
        counter.update([word for word in sent.split()])
    lexicon = sorted(counter)
    lexicon.insert(0, (
        lexicon.pop(lexicon.index("<unk>"))
        if "<unk>" in lexicon else "<unk>"
    ))

    filename = f"{output_dir}/{subset}_word_units.txt"
    with open(filename, 'w') as f:
        for i, word in enumerate(lexicon, 1):
            f.write(f"{word} {i}\n")

    print(f"Lexicon saved to {filename}")
    word2token = defaultdict(lambda: 1)
    word2token.update((word, i) for i, word in enumerate(lexicon, 1))
    return lexicon, word2token

lang_dir = "data/lang_word"
get_directory = "dump/{subset}_{tag}/deltafalse".format
input_json = "data.json"
for dataset in ["s", "m"]:

    vocab = word2token = None
    for subset in ["train", "dev", "test"]:
        subset_dir = get_directory(subset=subset, tag=dataset)
        with open(f"{subset_dir}/{input_json}") as infile:
            metadata = json.load(infile)

        ids, samples = map(list, zip(
            *sorted(metadata["utts"].items(), key=lambda t: t[0])
        ))

        sentences = (sample["output"][0]["text"] for sample in samples)
        if vocab is None:
            sentences = list(sentences)
            vocab, word2token = create_vocab(
                sentences, langdir, f"{subset}_{dataset}"
            )

        for uttid, sentence in tqdm(zip(ids, sentences), total=len(ids), desc=f"{subset}_{dataset}"):
            words = sentence.split()
            metadata["utts"][uttid]["output"][0] = {
                "name": "target1",
                "shape": [len(words), len(vocab)],
                "text": sentence,
                "token": sentence,
                "tokenid": " ".join(map(str, map(word2token.get, words)))
            }

        newfile = f"{subset_dir}/data_words_{len(vocab)}.json"
        with open("newfile", "w") as outfile:
            json.dump(metadata, outfile)

        print(f"Dataset written to {newfile}")

# %save -r RAW_create_word_lexicon.py 1-115
