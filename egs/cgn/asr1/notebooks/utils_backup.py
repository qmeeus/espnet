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



def load_evaluation_results(model_dir, training_sets=["o", "ok", "mono", "all"], test_sets=list("abfghijklmno")):
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