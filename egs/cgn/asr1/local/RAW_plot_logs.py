#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["loss", "accuracy", "both"]
LOSS_NAMES = ["loss", "loss_att", "loss_ctc"]
METRIC_NAMES = ["accuracy"]
logfile = "exp/train_trf_mtlalpha0.1_pos_data_lg_long_utt/results/log"

def load_logs(logfile):

    def rename_columns(name):
        comps = name.split("/")
        if "validation" in comps: return comps[0], comps[-1]
        elif "main" in name: return "train", comps[-1]
        else: return "main", name

    with open(logfile) as f:
        data = json.loads(f'{{"results": {f.read()}}}')
    results = pd.DataFrame.from_dict(data["results"])
    results.columns = pd.MultiIndex.from_tuples(
        list(map(rename_columns, results.columns))
    )
    return results


def plot_epoch_summary(results, mode, fname=None, logy=False, figsize=(12, 12), ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if mode == "both":
        plot_epoch_summary(results, "loss", logy=logy, ax=ax)
        plot_epoch_summary(results, "accuracy", ax=ax.twinx())
    else:
        metrics = [LOSS_NAMES, METRIC_NAMES][mode != "loss"]
        epoch_summary = results.groupby(("main", "epoch")).mean()
        epoch_summary.index = epoch_summary.index.rename("epoch")

        columns = list(filter(lambda t: t[1] in metrics, epoch_summary.columns))
        print(columns, epoch_summary.columns)
        epoch_summary = epoch_summary[columns]
        epoch_summary.columns = ["_".join(col) for col in epoch_summary.columns]
        epoch_summary.plot(logy=logy, ax=ax)

    if fname:
        plt.tight_layout()
        plt.savefig(fname)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("logs", type=Path)
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--width", type=int, default=8)
    return parser.parse_args()


def main():
    options = parse_args()
    results = load_logs(options.logs)
    fname = f"exp/graphs/{options.mode}_{options.logs.parent.name}.png"
    plot_epoch_summary(results, options.mode, fname, options.logy, (options.width, options.height))


if __name__ == "__main__":
    main()


"""
epoch_losses.plot()
plt.savefig("exp/graphs/losses.png")
np.log(epoch_losses).plot()
np.log(epoch_losses).plot()
plt.savefig("exp/graphs/log_losses.png")
epoch_losses.plot(log_y=True)
epoch_losses.plot(logy=True)
plt.savefig("exp/graphs/log_losses.png")
epoch_losses
epoch_losses.plot(logy=True)
plt.gca().set_xlabel("")
plt.savefig("exp/graphs/log_losses.png")
plt.tight_layout()
plt.savefig("exp/graphs/log_losses.png")
%save -r local/RAW_plot_logs.py 1-105
"""
