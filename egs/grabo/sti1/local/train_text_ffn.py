# coding: utf-8
import os
import numpy as np
import pandas as pd
import json
import torch
import functools
import multiprocessing as mp
from pathlib import Path
from operator import getitem
from sklearn import metrics
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import TensorDataset, Subset, DataLoader
import pytorch_lightning as pl


DATASETS = {
    "grabo": {
        "text": "data/grabo/text",
        "target": "data/grabo_w2v_encoded_target.csv"
    },
    "patience": {
        "text": "../../patience/sti1/data/text",
        "target": "../../patience/sti1/data/encoded_target.csv"
    }
}


class IntentClassifier(pl.LightningModule):

    def __init__(self, input_shape, output_shape, dropout_rate=.1):
        super(IntentClassifier, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.classifier = torch.nn.Linear(input_shape, output_shape)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.classifier(self.dropout(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        log = {"loss": loss, **self.compute_metrics(logits, y)}
        return loss, log

    def training_step(self, batch, batch_idx):
        loss, log = self.step(batch, batch_idx)
        results = pl.TrainResult(minimize=loss)
        results.log_dict({f"train_{k}": v for k, v in log.items()}, prog_bar=True)
        return results

    def validation_step(self, batch, batch_idx):
        loss, log = self.step(batch, batch_idx)
        results = pl.EvalResult()
        results.log_dict({f"val_{k}": v for k, v in log.items()})
        return results

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        log = {"loss": loss, **self.compute_metrics(logits, y)}
        results = pl.EvalResult()
        results.log_dict({f"test_{k}": v for k, v in log.items()})
        return results

    def compute_metrics(self, logits, target):
        y_true = target.long().cpu().numpy()
        y_pred = (torch.sigmoid(logits) > .5).long().cpu().numpy()
        f1_macro = metrics.f1_score(y_true, y_pred, average="macro")
        f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
        return {
            "f1_macro": torch.tensor(f1_macro),
            "f1_micro": torch.tensor(f1_micro)
        }


class GraboDataset(pl.LightningDataModule):

    INPUT_FILE = "data/grabo/text"
    TARGET_FILE =  "data/grabo_w2v/encoded_target.csv"

    def __init__(self, input_file=None,
                 target_file=None,
                 combine="average",
                 batch_size=512):

        super(GraboDataset, self).__init__()
        input_file = input_file or self.INPUT_FILE
        target_file = target_file or self.TARGET_FILE
        self.batch_size = batch_size
        self.combine = combine
        self.target_names = self.indices = None
        X, y  = self.load_data(input_file, target_file)
        self.data = TensorDataset(X, y)
        self.index_map = {index: i for i, index in enumerate(self.indices)}
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

    def get_subset(self, indices):
        ids = [self.index_map[index] for index in indices if index in self.index_map]
        return Subset(self.data, ids)

    def get_data_loader(self, indices):
        return DataLoader(self.get_subset(indices), batch_size=self.batch_size)

    def load_data(self, input_file, target_file):
        target = pd.read_csv(target_file, index_col="uttid")
        self.target_names = list(target.columns)
        self.indices = target.index

        if input_file.endswith("h5"):
            import h5py
            with h5py.File(input_file, "r") as h5f:
                self.indices = list(filter(lambda idx: idx in h5f, self.indices))
                features = [torch.tensor(h5f[index][()]) for index in self.indices]
        else:
            with open(input_file) as f:
                texts = pd.DataFrame(
                    map(functools.partial(str.split, maxsplit=1),
                        map(str.strip, f.readlines())),
                    columns=["uttid", "text"]
                ).set_index("uttid").dropna(how='any')
            self.indices = list(filter(lambda idx: idx in texts.index, self.indices))
            texts = texts.loc[self.indices].copy()
            features = self.encode_texts(texts)

        target = target.loc[self.indices]
        features = torch.stack([
            torch.mean(feats, dim=0) if self.combine == "average" else feats[0]
            for feats in features
        ], dim=0)

        target = torch.tensor(target.values).float()
        return features, target

    @staticmethod
    def encode_texts(texts, return_list=True):
        model_string = "distilbert-base-multilingual-cased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_string)
        model = DistilBertModel.from_pretrained(model_string)
        for param in model.parameters():
            param.requires_grad = False

        encoded_texts = tokenizer.batch_encode_plus(
            texts["text"], add_special_tokens=False, padding=True, return_tensors="pt"
        )
        features, = model(**encoded_texts)
        if return_list:
            input_lengths = encoded_texts["attention_mask"].sum(-1)
            return [features[i, :j] for i, j in enumerate(input_lengths)]
        return features



class PatienceDataset(GraboDataset):

    INPUT_FILE = "../../patience/sti1/data/text"
    TARGET_FILE = "../../patience/sti1/data/encoded_target.csv"


def train(speaker_dir, exp_name, input_file=None, dataset="grabo", max_epochs=100):


    Dataset = GraboDataset if dataset == "grabo" else PatienceDataset
    dataset = Dataset(input_file=input_file)
    for exp_dir in Path(speaker_dir).glob("*blocks_exp*"):
        outdir = Path(exp_dir, exp_name)
        if (outdir / "test_scores.txt").exists(): continue
        train_ids, test_ids = (np.loadtxt(f"{exp_dir}/{split}.ids", dtype="str") for split in ("train", "test"))
        train_ids, test_ids = (list(filter(lambda id: id in dataset.indices, ids)) for ids in (train_ids, test_ids))
        train_loader, test_loader = (dataset.get_data_loader(ids) for ids in (train_ids, test_ids))
        model = IntentClassifier(dataset.input_dim, dataset.output_dim)
        trainer = pl.Trainer(max_epochs=max_epochs, gpus=1)
        trainer.fit(model, train_loader, test_loader)

        os.makedirs(outdir, exist_ok=True)
        for subset, loader, indices in [("train", train_loader, train_ids), ("test", test_loader, test_ids)]:
            try:
                (X, y), = iter(loader)
            except:
                with open("failed.txt", "a" if Path("failed.txt").exists() else "w") as f:
                    f.write(f"{outdir}\n")
                return
            y_hat = torch.sigmoid(model(X))
            y_pred = (y_hat > .5).long().cpu()

            with open(outdir/f"{subset}_scores.txt", "w") as f:
                f.write(metrics.classification_report(
                    y, y_pred, target_names=dataset.target_names
                ))

            #(pd.DataFrame(y_hat, index=indices, columns=dataset.target_names)
            # .to_pickle(f"{outdir}/{subset}_predictions.pkl"))


def map_fn(options):
    train(**options)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("speaker_dirs", nargs="+")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["grabo", "patience"], default="grabo")
    parser.add_argument("--njobs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)

    options = vars(parser.parse_args())
    njobs = options.pop("njobs")
    jobs = []
    for speaker_dir in options.pop("speaker_dirs"):
        jobs.append(options.copy())
        jobs[-1]["speaker_dir"] = speaker_dir

    if njobs == 1:
        train(**jobs[0])
    else:
        with mp.Pool(njobs) as pool:
            pool.map(map_fn, jobs)


