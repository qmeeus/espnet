import re
import json
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display 

import torch
from torch import nn

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16,7)
pd.set_option("display.float_format", "{:.4f}".format)

# class ModelLogs:

#     def __init__(self, logdir):
#         self.logdir = logdir
#         self.logs = self.load_training_log()

#     def load_training_log(self):
#         jsonfile = Path(self.logdir, "log")
#         return pd.read_json(jsonfile)

#     def plot_training(self, metric="loss", ax=None):
#         logs = self.logs
        
#         metrics = train_metric, val_metric = [
#             tmp.format(metric) for tmp in ("main/{}", "validation/main/{}")
#         ]
        
#         assert all(m in logs.columns for m in metrics)

#         ax = ax or plt.subplot()

#         logs[train_metric].plot(label="train loss (batch)", legend=False, ax=ax)
#         epoch_mask = logs[val_metric].notnull()


class ModelSummary:
    
    def __init__(self, model_dir, curriculum=True, evaluation=True):
        self.model_dir = model_dir
        self.curriculum = curriculum
        self.training_log = self.load_curriculum() if curriculum else self.load_training()
        self._evaluate = evaluation
        self.evaluation_log = self.load_evaluation()
        self.paths, self.uttids = self.load_evaluation_dump()
        
    def load_training(self):
        jsonfile = Path(self.model_dir, "train", "results", "log")
        return pd.read_json(jsonfile)
        
    def load_curriculum(self):
        tags = ["o", "ok", "mono", "all"]
        results = pd.DataFrame()
        for tag in tags:
            jsonfile = Path(self.model_dir, "train", tag, "results", "log")
            if not jsonfile.exists():
                print(f"{jsonfile} not found")
                continue
            df = pd.read_json(jsonfile)
            df["dataset"] = tag
            results = pd.concat([results, df], axis=0)
            print(f"{tag}: {len(df)} results")

        return (
            results
            .assign(new_dataset=(results.index == 0))
            .reset_index(drop=True)
        )

    def training_summary(self, ax=None, metric="loss"):
        training_log = self.training_log
        ax = ax or plt.subplot()
        training_log[f"main/{metric}"].plot(label="train loss (batch)", legend=False, ax=ax)
        epoch_mask = training_log[f"validation/main/{metric}"].notnull()

        keys = (["dataset"] if self.curriculum else []) + ["epoch"]        
        (training_log.reset_index().groupby(keys, as_index=False)
         .agg({f"main/{metric}": "mean", "index": "max"})
         .set_index("index").sort_index()
         .drop(keys, axis=1)
         .plot(ax=ax, label=f"{metric} (epoch)", legend=False, alpha=.8))

        (training_log.loc[epoch_mask, f"validation/main/{metric}"].dropna(how="any")
         .plot(ax=ax, label=f"val {metric} (epoch)", legend=False))

        ax.legend()
        if self.curriculum:
            for xi in training_log[training_log["new_dataset"]].index[1:]:
                ax.axvline(xi, color='gray')
            
    def load_evaluation(self):
        if not self.check_eval():
            print("Model was not evaluated?")
            return
        tags = list("abfghijklmno")
        results = pd.DataFrame()
        for tag in tags:
            jsonpath = Path(self.model_dir, "evaluate", tag, "results", "results.json")
            if not jsonpath.exists():
    #             print(f"{jsonpath} not found")
                continue

            with open(jsonpath, "rb") as jsonfile:
                data = json.load(jsonfile)["results"]
            df = pd.DataFrame.from_dict(data, orient="index")
            df["dataset"] = tag
            results = pd.concat([results, df], axis=0)
    #         print(f"{tag}: {len(df)} results")
        
        return results
    
    def display_metrics(self):
        display(self.evaluation_log.groupby("dataset")
                .agg({"accuracy": "mean", "wer": "mean", "loss": "mean"}))
    
    def evaluation_summary(self):
        if self.check_eval():
            self.display_metrics()
            self.plot_random_attention()
        
    def check_eval(self):
        return self._evaluate and Path(self.model_dir, "evaluate").exists()
    
    def raise_on_no_eval(self):
        if not(self.check_eval):
            raise ValueError("Model was not evaluated?")

    def load_evaluation_dump(self):
        if not self.check_eval():
            print("Model was not evaluated?")
            return None, None
        
        paths = pd.DataFrame(
            list(Path(self.model_dir, "evaluate").glob("?/results/dump/*.npy")), 
            columns=["path"]
        )

        paths["comp"] = paths["path"].map(lambda p: p.parents[2].name)
        paths["type"] = paths["path"].map(lambda p: p.stem.split(".")[0])
        paths["batch_id"] = paths["path"].map(lambda p: p.stem.split(".")[1])
        paths = paths.set_index(["comp", "batch_id", "type"]).unstack(level=-1).droplevel(axis=1, level=0)

        uttids = pd.DataFrame((
            (uttid, comp, batch_id, sample_id) 
            for _, comp, batch_id, uttids in paths["uttids"].map(np.load).reset_index().itertuples() 
            for sample_id, uttid in enumerate(uttids)
        ), columns=["uttid", "comp", "batch_nr", "sample_id"]).set_index("uttid")
        
        return paths, uttids

    def plot_random_attention(self, uttid=None):
        self.raise_on_no_eval()
        sample = self.uttids.sample(1).iloc[0]
        attn = np.load(self.paths.loc[tuple(sample)[:2], "attn_ws"])[int(sample["sample_id"])]
        ax = sns.heatmap(attn, cmap='viridis')
        ax.set_title(f"{sample.name} (comp-{sample.comp})")
        
    def display(self):
        self.training_summary()
        self.evaluation_summary()


def load_logs(*model_logs, parse_model=None):
    n_models = len(model_logs)
    logs = pd.DataFrame()
    for logfile in model_logs:
        logfile = Path(logfile)
        log = pd.read_json(logfile)
        log["model_id"] = logfile.parents[2]
        logs = pd.concat([logs, log], axis=0)
        
    if parse_model is not None:
        logs = (
            pd.concat([logs, logs["model_id"].map(str).str.extract(parse_model)], axis=1)
        )

    return logs

def plot_comparison(results, groups, criterion="validation/main/loss", ascending=False, nbests=0):

    graph_keys = {
        "Loss": ["main/loss", "validation/main/loss"],
        "Attention loss": ["main/loss_att", "validation/main/loss_att"],
        "CTC loss": ["main/loss_ctc", "validation/main/loss_ctc"],
        "Accuracy": ["main/accuracy", "validation/main/accuracy"],
        "CER": ["main/cer_ctc", "validation/main/cer_ctc"],
        "Time": ["epoch", "elapsed_time"],
    }
    
    best_models = (
        results.sort_values(criterion, ascending=ascending)
        .groupby("model_id").head(1)
        .sort_values(criterion, ascending=ascending)
        .pipe(lambda df: df.head(nbests) if nbests else df)
        .set_index(groups)
        .drop([
            "model_id", "eps", "main/wer", "main/cer", "validation/main/wer", 
            "validation/main/cer", "iteration"], axis=1)
        .droplevel([0,1])
#         .assign(elapsed_time=lambda df: df.elapsed_time / 100)
    )
    
    fig, axs = plt.subplots(3, 2, figsize=(18,12), sharex=True)

    for i, (title, keys) in enumerate(graph_keys.items()):
        ax = axs[i%3, i//3]
        ax = best_models[keys].plot.bar(ax=ax, legend=False, secondary_y=keys[1] if title=="Time" else None)
        ax.set_title(title)
        ax.legend(loc='best')

def plot_training(results, criterion="validation/main/loss", ascending=False, nbests=0):

    graph_keys = {
        "Loss": ["main/loss", "validation/main/loss"],
        "Attention loss": ["main/loss_att", "validation/main/loss_att"],
        "CTC loss": ["main/loss_ctc", "validation/main/loss_ctc"],
        "Accuracy": ["main/accuracy", "validation/main/accuracy"],
        "CER": ["main/cer_ctc", "validation/main/cer_ctc"],
    }
    
    epoch_results = (
        results.assign(model_id=lambda df: df.model_id.map(lambda p: p.name))
        .groupby(["model_id", "epoch"]).mean()
        .drop(["eps", "elapsed_time", "main/wer", "main/cer", 
               "validation/main/wer", "validation/main/cer", "iteration"], axis=1)
    )

    if nbests > 0:
        model_ids = (epoch_results.sort_values(criterion, ascending=ascending)
                     .groupby(level=0).head(1).head(nbests).droplevel(1).index)
    else:
        model_ids = epoch_results.index.levels[0].unique()
    
    palette = sns.color_palette(n_colors=len(model_ids))

    fig, axs = plt.subplots(3, 2, figsize=(18,18), sharex=True)

    for i, (title, keys) in enumerate(graph_keys.items()):
        ax = axs[i%3, i//3]

        for i, model_id in enumerate(model_ids):
            (epoch_results.xs(model_id, level=0, axis=0)[keys]
             .plot(ax=ax, legend=False, color=[palette[i]] * len(keys)))

        ax.set_title(title)

    handles, _ = ax.get_legend_handles_labels()
    ncol = min(1, len(model_ids) // 3)
    ax.legend(handles[::2], model_ids, bbox_to_anchor=(.7 + ncol / 10, -.4), ncol=ncol)
    axs[-1,-1].remove()


def get_evaluation_results(model_dir, training_sets=["o", "ok", "mono", "all"], test_sets=list("abfghijklmno")):
    results_json = "{model_dir}/train/{train}/evaluate/results.{test}.json"
    results = pd.DataFrame() 
    for train_set in training_sets: 
        for test_set in test_sets: 
            jsonfile =results_json.format(model_dir=model_dir, train=train_set, test=test_set)           
            if not Path(jsonfile).exists(): 
                print(f"{jsonfile} not found") 
                continue 
            
            with open(jsonfile) as f: 
                df = pd.DataFrame.from_dict(json.load(f)["utts"], orient="columns") 
                df["train_set"] = train_set 
                df["test_split"] = test_set 
                results = pd.concat([results, df], axis=0)

    return results
