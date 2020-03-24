#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import kaldiio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class Visualiser:

    @staticmethod
    def parse_args(parser):
        parser.add_argument("dataset", type=Path, help="Path to json dataset")
        parser.add_argument("model_dir", type=Path, help="Path to model folder")
        parser.add_argument("--output_dir", type=Path, default="exp/graphs")

    def __init__(self, model_dir:Path, dataset:Path, output_dir:Path):
        self.model_dir = model_dir
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(dataset) as jsonfile:
            self.metadata = json.load(jsonfile)["utts"]

        self._tgtphr = self.metadata[self.get_random_uttid()]["output"][0]["token"][0]

    def get_random_uttid(self):
        uttids = list(set(map(self.path2uttid, self.model_dir.glob("results/att_ws/*.npy"))))
        return np.random.choice(uttids)

    def get_last_epoch(self, uttid):
        epochs = sorted(set(self.model_dir.glob(f"results/att_ws/{uttid}*.npy")))
        return max(map(self.path2epoch, epochs))

    @staticmethod
    def path2uttid(path:Path):
        return ".".join(path.stem.split(".")[:-2])

    @staticmethod
    def path2epoch(path:Path):
        return int(path.stem.split(".")[-1])

    def _plot_features(self, uttid, ax=None, aspect="equal", cmap=None):
        feats = kaldiio.load_mat(self.metadata[uttid]["input"][0]["feat"])
        ax = ax or plt.subplot()
        ax.imshow(feats.T, aspect=aspect, cmap=cmap)
        ax.axis('off')
        return ax

    def plot_features(self, uttid:str, aspect="equal", cmap=None, fname="features.png"):
        self._plot_features(uttid, aspect=aspect, cmap=cmap)
        self.save_fig(fname)

    def _plot_attention(self, weights:Path, target:str, ax=None, aspect="equal", cmap=None):
        attn_ws = np.load(weights)
        target = target.replace(self._tgtphr, "").split(" ")
        assert len(target) == attn_ws.shape[0]
        ax = ax or plt.subplot()
        ax.imshow(attn_ws, aspect=aspect, cmap=cmap)
        ax.set_xticks([]); ax.set_xticklabels([])
        ax.set_yticks(np.arange(len(target)))
        ax.set_yticklabels(["<sos>"] + target)
        ax.tick_params(length=0., labelsize=16)
        return ax

    def plot_attention(self, uttid, epoch="last", plot_features=True, cmap=None, fname="attention_weights.png"):
        if plot_features:
            fig = plt.figure(figsize=(12,12))
            gs = fig.add_gridspec(4,4)
            ax_attn = fig.add_subplot(gs[1:, :])
            ax_feats = fig.add_subplot(gs[0, :])
            self._plot_features(uttid, ax=ax_feats, aspect="auto", cmap=cmap)
        else:
            ax_attn = plt.subplot()
        epoch = epoch if type(epoch) is int else self.get_last_epoch(uttid)
        weights = Path(self.model_dir, "results", "att_ws", f"{uttid}.ep.{epoch}.npy")
        target = self.metadata[uttid]["output"][0]["token"]
        self._plot_attention(weights, target, ax=ax_attn, aspect='auto', cmap=cmap)
        self.save_fig(fname)

    def save_fig(self, fname):
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{fname}")
        plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plot", type=str,
                        choices=["feats", "attention", "metrics"],
                        help="What do we plot?")
    parser.add_argument("--uttid", "-u", type=str, default=None)
    Visualiser.parse_args(parser)
    options = vars(parser.parse_args())
    plot = options.pop("plot")
    uttid = options.pop("uttid")
    visualiser = Visualiser(**options)
    uttid = uttid or visualiser.get_random_uttid()

    if plot == "feats":
        visualiser.plot_features(uttid)
    elif plot == "attention":
        visualiser.plot_attention(uttid)
    else:
        raise NotImplemented


def test_plot_attention():
    uttid = "V40244-fv400782.85"
    #dataset = Path("dump/dev_m/deltafalse/data_pos_300.lg.150+.json")
    dataset = Path("dump/dev_m/deltafalse/data_unigram_1000.lg.150+.json")
    unigram_1000 = Path("exp/train_trf_mtlalpha0.1_unigram_1000_data_lg_long_utt")
    visualiser = Visualiser(unigram_1000, dataset, Path("exp/graphs"))
    visualiser.plot_attention(uttid)


if __name__ == '__main__':
    main()


# %save -r local/RAW_plot_attention.py 1-139
