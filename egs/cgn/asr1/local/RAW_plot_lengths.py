# coding: utf-8
datasets = ["o", "ok", "mono", "all"]
subsets = ["CGN_train", "CGN_valid"]
def plot_length_dist(jsonpath, ax=None, label=""):
    ax = ax or plt.subplot(111)
    with open(jsonpath, "rb") as jsonfile:
        dataset = json.load(jsonfile)["utts"]
    output_lengths = list(map(get_output_length, dataset.values()))
    sns.distplot(output_lengths, kde=True, ax=ax, label=label)
    return ax
    
def get_output_length(sample):
    return int(sample["output"][0]["shape"][0])
    
def plot_length_dist(jsonpath, ax=None, label=""):
    ax = ax or plt.subplot(111)
    with open(jsonpath, "rb") as jsonfile:
        dataset = json.load(jsonfile)["utts"]
    output_lengths = list(map(get_output_length, dataset.values()))
    N = len(output_lengths)
    mean = sum(output_lengths) / N
    std = math.sqrt(sum(l**2 for l in output_lengths) / N - mean **2)
    label = f"{label} (N={N:,} $\\bar{{x}}={mean:.1f}$ $\\sigma={std:.1f}$)"
    sns.distplot(output_lengths, kde=True, ax=ax, label=label)
    return ax
    
    
fig, axs = plt.subplots(2, 1, figsize=(18,12))
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(18,12))
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset}")
    ax.legend()
    ax.set_title(subset)
    
import json
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset}")
    ax.legend()
    ax.set_title(subset)
    
import math
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset}")
    ax.legend()
    ax.set_title(subset)
    
import seaborn as sns
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset}")
    ax.legend()
    ax.set_title(subset)
    
plt.savefig("exp/graphs/length_dist.png")
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset:>4}")
    ax.legend()
    ax.set_title(subset)
    
    
plt.close("all")
fig, axs = plt.subplots(2, 1, figsize=(18,12))
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset:>4}")
    ax.legend()
    ax.set_title(subset)
    
    
plt.tight_layout()
plt.savefig("exp/graphs/length_dist.png")
plt.close("all")
fig, axs = plt.subplots(2, 1, figsize=(18,12))
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset:4}")
    ax.legend()
    ax.set_title(subset)
    
plt.tight_layout()
plt.savefig("exp/graphs/length_dist.png")
plt.close("all")
fig, axs = plt.subplots(2, 1, figsize=(18,12))
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset}\t")
    ax.legend()
    ax.set_title(subset)
    
plt.tight_layout()
plt.savefig("exp/graphs/length_dist.png")
def plot_input_length_dist(jsonpath, ax=None, label=""):
    ax = ax or plt.subplot(111)
    with open(jsonpath, "rb") as jsonfile:
        dataset = json.load(jsonfile)["utts"]
    lengths = list(map(lambda s: s["input"][0]["shape"][0], dataset.values()))
    N = len(lengths)
    mean = sum(lengths) / N
    std = math.sqrt(sum(l**2 for l in lengths) / N - mean **2)
    label = f"{label} (N={N:,} $\\bar{{x}}={mean:.1f}$ $\\sigma={std:.1f}$)"
    sns.distplot(lengths, kde=True, ax=ax, label=label)
    return ax
    
    
    
fig, axs = plt.subplots(2, 1, figsize=(18,12))
for subset, ax in zip(subsets, axs):
    for dataset in datasets:
        plot_input_length_dist(f"dump/{subset}/deltafalse/data_words.{dataset}.json", ax=ax, label=f"{dataset}\t")
    ax.legend()
    ax.set_title(subset)
    
plt.tight_layout()
plt.savefig("exp/graphs/input_lengths.png")
%save -r local/RAW_plot_lengths.py 1-38
