import argparse
import json
import os
import torch
import pandas as pd
from pathlib import Path
import h5py

"""
Usage:
    python local/RAW_encode_sentences.py data/CGN_ALL/text \
        --gpu 0 --jsonfiles dump/CGN_*/deltafalse/data_unigram_25000.*.json

    python local/RAW_encode_sentences.py data/CGN_ALL/text.raw \
        --gpu 0 --jsonfiles dump/CGN_*/deltafalse/data_unigram_25000.*.json \
        --pretrained-model wietsedv/bert-base-dutch-cased --loader huggingface_bert \
        --output-file vectors/target_vectors_bert_dutch.h5
"""


class Encoder:

    # LOADER = "hub"
    # PRETRAINED_MODEL =  f"pytorch/fairseq/xlmr.base"
    # OUTPUT_FILE = Path(f"vectors/target_vectors_xlmr_base.h5").absolute()
    # JSON_PREFIX = "data_vectors_xlmr"

    LOADER = "huggingface_bert"
    MODEL_BASE_CLASS = "BertModel"
    PRETRAINED_MODEL =  f"wietsedv/bert-base-dutch-cased"
    OUTPUT_FILE = Path(f"vectors/target_vectors_bert_dutch.h5").absolute()
    JSON_PREFIX = "data_vectors_bert_dutch"

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("input_texts", type=Path)
        parser.add_argument("--pretrained-model", default=cls.PRETRAINED_MODEL)
        parser.add_argument("--loader", default=cls.LOADER)
        parser.add_argument("--model-base-class", default=cls.MODEL_BASE_CLASS)
        parser.add_argument("--gpu", type=int, default=None)
        parser.add_argument("--jsonfiles", type=Path, nargs="+", required=True)
        parser.add_argument("--json-prefix", type=str, default=cls.JSON_PREFIX)
        parser.add_argument("--output-file", type=Path, default=cls.OUTPUT_FILE)
        options = vars(parser.parse_args())
        input_texts = options.pop("input_texts")
        return cls(**options)(input_texts)

    def __init__(self, jsonfiles,
                 pretrained_model=PRETRAINED_MODEL,
                 loader=LOADER,
                 model_base_class=MODEL_BASE_CLASS,
                 json_prefix=JSON_PREFIX,
                 output_file=OUTPUT_FILE,
                 gpu=None):

        self.loader = loader
        self.model_base_class = model_base_class
        self.device = f'cuda:{gpu}' if gpu is not None else 'cpu'
        # self.model = self.load_pretrained_model(pretrained_model)
        self.tokenizer, self.model = self.load_pretrained_model(pretrained_model)
        self.configure_model(self.model, self.device)
        self.jsonfiles = jsonfiles
        self.json_prefix = json_prefix
        self.output_file = output_file
        os.makedirs(self.output_file.parent, exist_ok=True)

    def load_pretrained_model(self, model_string):
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_string)
        model = getattr(transformers, self.model_base_class).from_pretrained(model_string)
        return tokenizer, model


    def _load_pretrained_model(self, model_string):
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(model_string)
        model = BertModel.from_pretrained(model_string)
        return tokenizer, model

    def load_sentences_from_file(self, filename):
        with open(filename) as txtfile:
            yield from map(
                lambda t: (t[0], self.extract_features(t[1])), map(
                    lambda s: s.split(" ", maxsplit=1),
                    map(str.strip, txtfile.readlines())
                )
            )

    def __call__(self, sentences):

        shapes = {}
        with h5py.File(self.output_file, 'w') as h5_file:
            for uttid, feats in self.load_sentences_from_file(sentences):
                h5_file.create_dataset(uttid, data=feats)
                shapes[uttid] = feats.shape

        for filename in self.jsonfiles:

            parent, name = filename.parent, filename.name
            new_file = parent / ".".join([self.json_prefix, *name.split(".")[1:]])

            with open(filename) as old, open(new_file, 'w') as new:
                json.dump({"utts": dict(map(
                    self.update_item(shapes),
                    json.load(old)["utts"].items()
                ))}, new, indent=4)

    def update_item(self, shapes):
        def _update_item(item):
            uttid, item = item
            output = item["output"][0]
            item["output"][0] = {
                "name": "target1",
                "text": item["output"][0]["text"],
                "feat": f"{self.output_file}:{uttid}",
                "filetype": "hdf5",
                "shape": shapes[uttid]
            }
            return uttid, item
        return _update_item

    def encode(self, sentence):
        return torch.tensor(self.tokenizer.encode(sentence))

    def extract_features(self, sentence):
        encoded_sentence = self.encode(sentence).to(self.device)
        if encoded_sentence.ndim < 2:
            encoded_sentence = encoded_sentence.view(1, -1)

        with torch.no_grad():
            return self.model(encoded_sentence)[0].detach().cpu().squeeze(0).numpy()

    @staticmethod
    def configure_model(model, device):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(device)

    @staticmethod
    def batches(iterable, bs=128):
        l = len(iterable)
        for i in range(0, l, bs):
            yield iterable[i:min(l, i + bs)]

    ### BACKUP PYTORCH'S HUB

    # def load_pretrained_model(self, model_string):
    #     model = torch.hub.load(*self.parse_model_string(model_string))
    #     model.eval()
    #     model.to(self.device)
    #     return model

    # @staticmethod
    # def parse_model_string(model_string):
    #     parts = model_string.split("/")
    #     return "/".join(parts[:-1]), parts[-1]

    # def extract_features(self, sentence):
    #     encoded_sentence = self.encode(sentence).to(self.device)

    #     with torch.no_grad():
    #         return self.model.extract_features(encoded_sentence).detach().cpu().squeeze(0).numpy()

    # @property
    # def dictionary(self):
    #     return self.model.task.source_dictionary

    # def encode(self, sentence):
    #     # torch.hub, no eos/sos token
    #     return self.dictionary.encode_line(
    #         self.model.bpe.encode(sentence),
    #         append_eos=False,
    #         add_if_not_exist=False
    #     ).long()

    # def encode(self, sentence):
    #     # torch.hub
    #     return self.model.encode(sentence)



if __name__ == "__main__":
    Encoder.parse_args()

