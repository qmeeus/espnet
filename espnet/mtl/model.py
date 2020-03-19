import logging
import torch
import pytorch_lightning as pl
from copy import deepcopy
from itertools import chain
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import Adam, Adadelta


class ASRModel(pl.LightningModule):

    def __init__(self, model, datasets, args):
        super(ASRModel, self).__init__()
        self.model = model
        assert all(subset in datasets for subset in ("train", "validation"))
        self.datasets = datasets

        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.num_workers = args.num_workers
        self.optimizer_name = args.optimizer_name
        self.optimizer_config = args.optimizer_config

    def forward(self, frames, frame_lengths, targets):
        return self.model(frames, frame_lengths, targets)


    def training_step(self, batch, batch_idx):
        frames, frame_lengths, targets, target_lengths = batch
        outputs = self.forward(frames, frame_lengths, targets)
        losses = self.model.compute_loss(outputs, targets, target_lengths)
        
        if self.trainer.use_dp or self.trainer.use_ddp2:
            losses = tuple(tensor.unsqueeze(0) for tensor in losses)

        tqdm_dict = dict(
            zip(map(lambda s: f"train_{s}", self.model.LOSS_NAMES), losses)
        )

        loss, loss_ctc, loss_att = losses

        return OrderedDict(
            loss=loss,
            loss_ctc=loss_ctc, 
            loss_att=loss_att,
            progress_bar=tqdm_dict,
            log=tqdm_dict
        )

    def validation_step(self, batch, batch_idx):
        frames, frame_lengths, targets, target_lengths = batch
        outputs = self.forward(frames, frame_lengths, targets)
        losses = self.model.compute_loss(outputs, targets, target_lengths)
        metrics = self.model.compute_metrics(outputs, targets, target_lengths)
        
        if self.trainer.use_dp or self.trainer.use_ddp2:
            losses = (tensor.unsqueeze(0) for tensor in losses)
            metrics = (tensor.unsqueeze(0) for tensor in metrics)

        tqdm_dict = dict(chain(
            zip(map(lambda s: f"val_{s}", self.model.LOSS_NAMES), losses), 
            zip(map(lambda s: f"val_{s}", self.model.METRIC_NAMES), metrics)
        ))
        
        return OrderedDict(
            **tqdm_dict,
            progress_bar=tqdm_dict,
            log=tqdm_dict,
        )
        
    def validation_epoch_end(self, outputs):
        if not outputs:
            return {}

        metric_names = self.model.LOSS_NAMES + self.model.METRIC_NAMES

        metrics = {
            metric_name: torch.stack([x[metric_name] for x in outputs]).mean()
            for metric_name in map(lambda s: f"val_{s}", metric_names)
            if metric_name in outputs[0]
        }

        return {'progress_bar': metrics, 'log': metrics, 'val_loss': metrics["val_loss"]}

    def format_logs(self, losses, metrics):
        logs = OrderedDict()
        logs.update(losses)
        logs.update(metrics)
        progress_bar = dict(filter(lambda t: t[0] != 'loss', losses.items()))
        logs.update({'progress_bar': progress_bar})
        logs.update({'log': dict(**losses, **metrics)})
        return logs

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("validation")

    def configure_optimizers(self):
        opt = self.optimizer_name
        if opt == "adadelta":
            Optimizer = Adadelta
        elif opt == "adam":
            Optimizer = Adam
        else:
            raise NotImplementedError(f"Unknown optimizer: {opt}")
        return Optimizer(self.parameters(), **self.optimizer_config)

    def dataloader(self, subset):
        if subset not in self.datasets:
            raise ValueError(f"Dataset {subset} is not loaded")

        dataset = self.datasets[subset]
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_samples
        )
