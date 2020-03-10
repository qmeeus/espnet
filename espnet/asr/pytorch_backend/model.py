from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam

from espnet.data.dataset import ASRDataset

class ASRModel(pl.LightningModule):

    def __init__(self, model_class, args):
        super(ASRModel, self).__init__()

        self.datasets = {}
        self.load_dataset(args.train_json, "train")
        self.load_dataset(args.valid_json, "validation")
        
        # HACK for compatibility
        input_dim = self.datasets["train"].input_dim[0]
        output_dim = self.datasets["train"].output_dim[0]

        self.model = model_class(input_dim, output_dim, args)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.shuffle = getattr(args, 'shuffle', True)
        self.num_workers = getattr(args, 'num_workers', 10)
        self.optimizer_name = getattr(args, 'opt', 'adadelta')
        self.optimizer_config = {
            'rho': getattr(args, 'rho', None),
            'eps': getattr(args, 'eps', None),
            'weight_decay': getattr(args, 'weight_decay', None)
        }

    def forward(self, *inputs, **kwargs):
        return self.model.forward(*inputs, compat_on=False, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch, calc_metrics=True)
        return loss

    def load_dataset(self, json_file, name=None):
        name = name or Path(json_file).stem
        self.datasets[name] = ASRDataset.from_json(json_file)
        return self.datasets[name]

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

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("validation")

    def get_optimizer(model, args):
        opt = self.optimizer_name
        if opt == "adadelta":
            return Adadelta(
                self.parameters(), 
                rho=self.optimizer_config["rho"], 
                eps=self.optimizer_config["eps"],
                weight_decay=self.optimizer_config["weight_decay"]
            )
        
        elif opt == "adam":
            return Adam(
                self.parameters(),
                weight_decay=self.optimizer_config["weight_decay"]
            )
        
        else:
            raise NotImplementedError(f"Unknown optimizer: {opt}")
        