import json
import logging
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py

from espnet.data.dataset import ASRDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.batchfy import make_batchset
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.utils.dynamic_import import dynamic_import
from espnet.asr.pytorch_backend.converter import CustomConverter
from espnet.asr.pytorch_backend.trainer import CustomTrainer
from espnet.utils.dataset import ChainerDataLoader, TransformDataset
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.torch_utils import _recursive_to
from espnet.nets.pytorch_backend.nets_utils import pad_list

from espnet.w2v.w2v import build_model, get_optimizer, setup, load_dataset, train


def recog(options):

    # SETUP AND PRELIMINARY OPERATIONS
    model, optimizer, converter, device = setup(options, train=False)

    with open(options.test_json, "rb") as json_file:
        metadata = json.load(json_file)["utts"]

    dataset = make_batchset(
        metadata, options.batch_size,
        options.maxlen_in, options.maxlen_out, options.minibatches,
        min_batch_size=options.ngpu if options.ngpu > 1 else 1,
        shortest_first=options.use_sortagrad if train else False,
        count=options.batch_count,
        batch_bins=options.batch_bins,
        batch_frames_in=options.batch_frames_in,
        batch_frames_out=options.batch_frames_out,
        batch_frames_inout=options.batch_frames_inout,
        iaxis=0, oaxis=0
    )

    loader = LoadInputsAndTargets(
        mode='asr',
        load_output=True,
        preprocess_conf=options.preprocess_conf,
        preprocess_args={'train': train}  # Switch the mode of preprocessing
    )

    def convert(data):
        X, y = (list(map(torch.tensor, ar)) for ar in loader(data))
        Xlens, ylens = (torch.tensor(list(map(len, tensor))) for tensor in (X, y))
        uttids = list(map(lambda example: example[0], data))

        X, y = (ASRDataset._collate(tensor, 0, 0) for tensor in (X, y))
        return {
            "input1": X, 
            "input1_length": Xlens, 
            "target1": y,
            "target1_length": ylens,
            "uttid": uttids
        }

    iterator = ChainerDataLoader(
        TransformDataset(dataset, convert),
        batch_size=1,
        shuffle=not(options.use_sortagrad) if train else False,
        num_workers=options.n_iter_processes,
        collate_fn=lambda x: x
    )

    dump_dir = f"{options.outdir}/dump"
    os.makedirs(dump_dir, exist_ok=True)

    with torch.no_grad(), h5py.File(f"{dump_dir}/predictions.h5", "w") as h5f:

        lengths = {}

        for i, (batch,) in enumerate(iterator):
            X, Xlens = batch["input1"], batch["input1_length"]
            y, ylens = batch["target1"], batch["target1_length"]            
            uttids = batch["uttid"]
            X, Xlens, y, ylens = _recursive_to((X, Xlens, y, ylens), device)

            loss, preds, pred_mask = model.predict(X, Xlens, y, ylens)
            pred_lengths = (~pred_mask[:, -1:, :]).sum(-1).squeeze(-1).tolist()

            for i, uttid in enumerate(uttids):
                h5f.create_dataset(
                    uttid, 
                    data=preds[i, :pred_lengths[i]].detach().cpu().numpy(), 
                    compression="gzip", compression_opts=9
                )
