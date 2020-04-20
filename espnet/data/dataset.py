import json
import numpy as np
from operator import itemgetter
from pathlib import Path

import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset
from kaldiio import load_mat

# TODO: implement caching
# TODO: see https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5


class ASRDataset(Dataset):
    """Transform Dataset for pytorch backend.

    Args:
        data: list object from make_batchset
        transform: transform function

    """
    def __init__(self, keys, metadata,
                 transform=None, 
                 cache_size=10,
                 features_pad_value=0, 
                 target_pad_value=-1, 
                 sort_by_length=True):

        super(ASRDataset).__init__()
        self.sort_by_length = sort_by_length

        self.keys = keys
        self.metadata = metadata
        self.input_lengths = list(map(lambda el: el["input"][0]["shape"][0], metadata))
        
        if sort_by_length:
            self.keys, self.metadata, self.input_lengths = (
                map(list, zip(*sorted(
                    zip(self.keys, self.metadata, self.input_lengths), 
                    key=lambda t: t[2], reverse=True
                )))
            )

        self.n_inputs = len(metadata[0]["input"])
        self.n_outputs = len(metadata[0]["output"])
        if not(self.n_inputs == self.n_outputs == 1):
            raise NotImplementedError("Only one inputs/outputs at the moment")

        self.input_dim, self.output_dim = ([
            metadata[0][k][i]["shape"][1] 
            for i in range(getattr(self, f'n_{k}s'))
        ] for k in ("input", "output"))
        
        self.data_cache = {}
        self.cache_size = cache_size
        self.transform = transform
        self.features_pad_value = features_pad_value
        self.target_pad_value = target_pad_value

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        metadatum = self.metadata[idx]
        key = self.keys[idx]
        sample = {"uttid": key}
        for input in metadatum["input"]:
            features = self.get_data(input["feat"], key)
            if self.transform is not None:
                features = self.transform(features)
            sample[input["name"]] = features
            sample[f"{input['name']}_length"] = features.size(0)
        for output in metadatum["output"]:
            tokenids = list(map(int, output["tokenid"].split()))
            sample[output["name"]] = torch.LongTensor(tokenids)
            sample[f"{output['name']}_length"] = len(tokenids)
        return sample

    def get_data(self, filepath, key):
        if filepath not in self.data_cache:
            self._load_data(filepath)
        return self.data_cache[filepath]

    def _load_data(self, filepath):
        if len(self.data_cache) > self.cache_size:
            to_remove = list(self.data_cache)[0]
            self.data_cache.pop(to_remove)
        self.data_cache[filepath] = torch.from_numpy(load_mat(filepath))

    def collate_samples(self, samples, maxlen=None):
        samples = {k: [sample[k] for sample in samples] for k in samples[0]}
        batch = {}
        for k, v in samples.items():
            if k.endswith("length"):
                batch[k] = torch.LongTensor(v)
            elif "input" in k:
                batch[k] = self._collate(v, 0, self.features_pad_value, maxlen)
            elif "target" in k:
                batch[k] = self._collate(v, 0, self.target_pad_value, maxlen)
            else:
                batch[k] = np.array(v)
                # raise ValueError(f"Unexpected element in batch: {k}")
        # espnet compatibility: return tuple instead of dict
        # batch = batch["input1"], batch["input1_length"], batch["target1"], batch["target1_length"]
        if self.sort_by_length:
            batch = self.sort_batch(batch, "input1_length")
        return batch

    def _collate(self, tensors, axis, padding_value, maxlen=None):
        maxlen = maxlen or max(tensor.size(axis) for tensor in tensors)
        size = [len(tensors), *tensors[0].size()]
        size[axis + 1] = maxlen
        axis = axis if axis >= 0 else len(size) + axis
        padded_tensors = torch.Tensor(*size).type_as(tensors[0])
        padding = [0] * ((len(size) - 1) * 2)
        padding_idx = tensors[0].dim() * 2 - (axis * 2 + 1)
        for i, tensor in enumerate(tensors):
            padding[padding_idx] = maxlen - tensor.size(axis)
            padded_tensors[i, :] = pad(tensor, padding, mode='constant', value=padding_value)
        return padded_tensors

    def sort_batch(self, batch, key="input1_length"):
        # batch is a dictionary
        Xlens = batch[key]
        assert torch.is_tensor(Xlens)
        Xlens, sort_order = Xlens.sort(descending=True)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.index_select(0, sort_order)
            elif type(v) == np.ndarray:
                batch[k] = v[sort_order]
            else:
                raise NotImplementedError(f"{k}: {type(v)}")
        return batch

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
                            "text": "luc had nog een buurman geïnviteerd",
                            "token": "l u c <space> h a d <space> n o g <space> e e n <space> b u u r m a n <space> g e ï n v i t e e r d",
                            "tokenid": "29 38 20 16 25 18 21 16 31 32 24 16 22 22 31 16 19 38 38 35 30 18 31 16 24 22 56 31 39 26 37 22 22 35 21"
                        }
                    ],
                    "utt2spk": "V40194"
                }
            ...
            }
        }
        kwargs are passed to the constructor
        """
        with open(path, 'rb') as json_file:
            metadata = json.load(json_file)
        if not metadata:
            raise TypeError("Empty JSON: {}".format(path))
        if "utts" not in metadata:
            raise TypeError("Malformed JSON: {}".format(path))
        keys, metadata = zip(*sorted(metadata["utts"].items(), key=itemgetter(0)))
        return cls(keys, metadata, **kwargs)
