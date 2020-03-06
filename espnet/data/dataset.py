import json
import numpy as np
from operator import itemgetter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchaudio.kaldi_io import read_mat_ark

# TODO: implement caching
# TODO: see https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5


class SpeechDataset(Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transform: transform function

    """
    def __init__(self, keys, metadata, transform=None, cache_size=10):
        """Init function."""
        super(SpeechDataset).__init__()
        self.keys = keys
        self.metadata = metadata
        self.data_cache = {}
        self.cache_size = cache_size
        self.transform = transform

    def __len__(self):
        """Len function."""
        return len(self.data)

    def __getitem__(self, idx):
        """[] operator."""
        features = self.get_data(idx)
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_data(self, idx):
        metadata = self.metadata[idx]
        key = self.keys[idx]
        filepath = metadata["input"]["feat"]
        if filepath not in self.data_cache:
            self._load_data(filepath)
        return self.data_cache["data"][filepath][key]

    def _load_data(self, filepath):
        if len(self.data_cache) > self.cache_size:
            to_remove = list(self.data_cache)[0]
            self.data_cache.pop(to_remove)
        self.data_cache[filepath] = dict(read_mat_ark(filepath))

    @classmethod
    def from_json(cls, path, **kwargs):
        """path: json input file e.g.
        {
            "utts": {
                "V40194-fv801122.6": {
                    "input": [
                        {
                            "feat": "/tmp/espnet/egs/cgn/asr1/dump/train_s_/deltafalse/feats.1.ark:18",
                            "name": "input1",
                            "shape": [390,83]
                        }
                    ],
                    "output": [
                        {
                            "name": "target1",
                            "shape": [62,65],
                            "text": "luc had nog een buurman geïnviteerd eddy die ik van naam kende",
                            "token": "l u c <space> h a d <space> n o g <space> e e n <space> b u u r m a n <space> g e ï n v i t e e r d <space> e d d y <space> d i e <space> i k <space> v a n <space> n a a m <space> k e n d e",
                            "tokenid": "29 38 20 16 25 18 21 16 31 32 24 16 22 22 31 16 19 38 38 35 30 18 31 16 24 22 56 31 39 26 37 22 22 35 21 16 22 21 21 42 16 21 26 22 16 26 28 16 39 18 31 16 31 18 18 30 16 28 22 31 21 22"
                        }
                    ],
                    "utt2spk": "V40194"
                }
            ...
            }
        kwargs are passed to the constructor
        """
        with open(path) as json_file:
            metadata = json.load(json_file)
        if not metadata:
            raise TypeError("Empty JSON: {}".format(path))
        if "utts" not in metadata:
            raise TypeError("Malformed JSON: {}".format(path))
        keys, metadata = zip(*sorted(metadata["utts"].items(), key=itemgetter(0)))
        return cls(keys, metadata, **kwargs)
