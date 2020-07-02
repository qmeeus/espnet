import argparse
import json
import os
import torch
import pandas as pd
from pathlib import Path


class Encoder:

    PRETRAINED_MODEL_NAME = "xlmr.base"
    PRETRAINED_MODEL =  f"pytorch/fairseq/{PRETRAINED_MODEL_NAME}"
    LANGDIR = Path(f"data/lang_{PRETRAINED_MODEL_NAME}")

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("input_texts", type=Path)
        parser.add_argument("--pretrained-model", default=cls.PRETRAINED_MODEL)
        parser.add_argument("--langdir", type=Path, default=cls.LANGDIR)
        options = vars(parser.parse_args())
        input_texts = options.pop("input_texts")
        return cls(**options).fit(input_texts)

    def __init__(self, pretrained_model, langdir):
        self.pretrained_model = self.parse_model_string(pretrained_model)
        self.langdir = langdir
        self.model = torch.hub.load(*self.pretrained_model)
        self.model.eval()
        os.makedirs(self.langdir, exist_ok=True)

    def load_sentences_from_file(self, filename):
        with open(filename) as txtfile:
            self.uttids_, self.sentences_ = map(list, zip(*map(
                lambda s: s.split(" ", maxsplit=1),
                map(str.strip, txtfile.readlines()[:10])
            )))
        return self.sentences_

    def fit(self, sentences):
        if isinstance(sentences, Path):
            sentences = self.load_sentences_from_file(sentences)
        encoded_sentences = list(map(torch.Tensor.tolist, map(self.encode, sentences)))
        unique_tokens = sorted(set(sum(encoded_sentences, [])))
        unique_tokens.insert(0, self.dictionary.pad())
        unique_tokens.append(self.dictionary.eos())
        mapping = {
            token_id: (self.dictionary[token_id], index)
            for index, token_id in enumerate(unique_tokens, 1)
        }

        with open(self.langdir / "mapping.json", "w", encoding="utf-8") as jsonfile:
            json.dump(mapping, jsonfile, indent=4)

        with open(self.langdir / "units.txt", "w", encoding="utf-8") as txtfile:
            txtfile.writelines([f"{sym} {index}\n" for sym, index in mapping.values()])

        import ipdb; ipdb.set_trace()
        vectors = torch.cat([
            self.extract(torch.tensor(batch)) for batch in self.batches(unique_tokens)
        ]).numpy()

        with open(self.langdir / "vectors.txt", "w") as vectfile:
            vectfile.write(f"{len(unique_tokens)} {vectors.shape[1]}\n")
            for i, token in enumerate(unique_tokens):
                vectfile.write(f"{mapping[token][0]} {' '.join(map(str, vectors[i]))}\n")

        encoded_sentences = list(map(
            lambda tokens: [mapping[token][1] for token in tokens], encoded_sentences
        ))

        # TODO: save vocabulary with BPE symbols and vectors
        # TODO: save tokenized sentences as json datasets

    @property
    def dictionary(self):
        return self.model.task.source_dictionary

    @property
    def encode(self):
        def _encode(sentence):
            bpe_sentence = self.model.bpe.encode(sentence)
            return self.dictionary.encode_line(
                bpe_sentence, append_eos=False, add_if_not_exist=False
            ).long()
        return _encode

    @property
    def extract(self):
        def _extract(tokens):
            with torch.no_grad():
                return self.model.extract_features(torch.tensor(tokens)).squeeze(0).numpy()
        return _extract

    @staticmethod
    def parse_model_string(model_string):
        parts = model_string.split("/")
        return "/".join(parts[:-1]), parts[-1]

    @staticmethod
    def batches(iterable, bs=128):
        l = len(iterable)
        for i in range(0, l, bs):
            yield iterable[i:min(l, i + bs)]

if __name__ == "__main__":
    Encoder.parse_args()

