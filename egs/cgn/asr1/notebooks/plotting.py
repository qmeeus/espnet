import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchaudio
import re
from pprint import pprint, pformat
from pathlib import Path
from IPython.display import Audio

from data import load_evaluation_results, annotations, cgn_root


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16,7)

__all__ = [
    "plt",
    "sns",
    "pprint",
    "plot_comparison",
    "plot_training",
    "plot_eval_results_by_length",
    "plot_attention",
    "plot_attention_predictions",
    "plot_attention_grid",
    "plot_attention_w2v_vs_ctc",
    "display_audio",
    "format_dataframe_best_metrics"
]


def plot_comparison(results):

    graph_keys = {
        "Loss": ["main/loss", "validation/main/loss"],
        "Attention loss": ["main/loss_att", "validation/main/loss_att"],
        "CTC loss": ["main/loss_ctc", "validation/main/loss_ctc"],
        "Accuracy": ["main/accuracy", "validation/main/accuracy"],
        "CER": ["main/cer_ctc", "validation/main/cer_ctc"],
        "Time": ["epoch", "elapsed_time"],
    }
        
    fig, axs = plt.subplots(3, 2, figsize=(18,12), sharex=True)

    for i, (title, keys) in enumerate(graph_keys.items()):
        ax = axs[i%3, i//3]
        ax = results[keys].plot.bar(ax=ax, legend=False, secondary_y=keys[1] if title=="Time" else None)
        ax.set_title(title)
        ax.legend(loc='best')

def plot_training(results, relative=False):

    graph_keys = {
        "Loss": ["main/loss", "validation/main/loss"],
        "Attention loss": ["main/loss_att", "validation/main/loss_att"],
        "CTC loss": ["main/loss_ctc", "validation/main/loss_ctc"],
        "Accuracy": ["main/accuracy", "validation/main/accuracy"],
        "CER": ["main/cer_ctc", "validation/main/cer_ctc"],
    }

    model_names = results["model_name"].unique()
    palette = sns.color_palette(n_colors=len(model_names))

    results = results.groupby(["model_name", "epoch"], as_index=False).mean().set_index("model_name")
    
    if relative:
        results["epoch"] = results["epoch"].groupby(level=0).apply(lambda g: g / g.max())
    
    fig, axs = plt.subplots(3, 2, figsize=(18,18), sharex=True)
    for i, (title, keys) in enumerate(graph_keys.items()):
        ax = axs[i%3, i//3]

        for i, model_name in enumerate(model_names):
            results.xs(model_name, axis=0).plot(
                x="epoch", 
                y=keys, 
                ax=ax, 
                legend=False, 
                color=[palette[i]] * len(keys)
            )
            

        ax.set_title(title)

    handles, _ = ax.get_legend_handles_labels()
    ncol = min(1, len(model_names) // 3)
    ax.legend(handles[::2], model_names, bbox_to_anchor=(.7 + ncol / 10, -.4), ncol=ncol)
    axs[-1,-1].remove()


def plot_eval_results_by_length(eval_results, metric, q=10, frac=.1):
    sns.swarmplot(
        x="test_split", y=metric, hue="len_cat", data=(
            eval_results
            .assign(len_cat=pd.qcut(eval_results["groundtruth"].str.split(" ").map(len), q))
            .sample(frac=frac)
        )
    )


def plot_attention(attention_weights, ax=None, title=None, **heatmap_options):
    options = dict(cbar=False, xticklabels=[], yticklabels=[], cmap="viridis")
    options.update(heatmap_options)
    options["ax"] = ax or plt.subplot()
    if isinstance(attention_weights, (Path, str)):
        attention_weights = np.load(attention_weights)
    attention_weights = pd.DataFrame(attention_weights)

#     if "xticklabels" in heatmap_options:
#         attention_weights.columns = heatmap_options.pop("xticklabels")
#     if "yticklabels" in heatmap_options:
#         attention_weights.index = heatmap_options.pop("yticklabels")
    
    ax = sns.heatmap(attention_weights, **options)
    ax.set_title(title or "")
    return ax


def plot_attention_predictions(attention_weights, predictions:dict, ax=None):
    ax = plot_attention(
        attention_weights, 
        xticklabels=["" if token == "<blank>" else token for token in predictions["ctc_raw_pred"]], 
        yticklabels=predictions["dec_raw_pred"],
        title=(
            "{groundtruth}\nCTC: cer={cer_ctc:.2%} ter={ter_ctc:.2%}  Decoder: cer={cer_dec:.2%} ter={ter_dec:.2%}"
            .format(**predictions)
        )
    )
    
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    

def plot_attention_grid(model_dir, test_splits=list("abfghijklmno")):
    template = "{model_dir}/evaluate/{test_split}/att_ws/{uttid}.npy"
    
    selection = (load_evaluation_results(model_dir)
                 .sample(frac=1.)
                 .groupby("test_split").head(1)
                 .sort_values("test_split"))
    
    nrows, ncols = len(test_splits) // 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    for i, (uttid, sample) in enumerate(selection.iterrows()):
        attention_weights = template.format(model_dir=model_dir, test_split=sample.test_split, uttid=uttid)
        ax = axs[i // ncols, i % ncols]
        plot_attention(attention_weights, ax=ax);
        ax.set_title(
            "comp-{test_split}: {uttid}\nCTC: cer={cer_ctc:.2%} ter={ter_ctc:.2%}\nDec: cer={cer_dec:.2%} ter={ter_dec:.2%}"
            .format(uttid=uttid, **sample.to_dict())
        )

    for j in range(i + 1, nrows * ncols):
        axs[j // ncols, j % ncols].remove()
        
    plt.tight_layout()

    
def plot_attention_w2v_vs_ctc(pretrained_model, decoder_model, test_splits=None, prob_scale=10, uttid=None):
    pre_model, dec_model = (Path(d) for d in (pretrained_model, decoder_model))
    pre_results, dec_results = (load_evaluation_results(d) for d in (pre_model, dec_model))

    if test_splits:
        dec_results = dec_results[dec_results.test_split.isin(test_splits)].copy()
    
    if uttid is None:        
        uttid = dec_results.sort_values("wer").iloc[int(np.random.exponential(prob_scale))].name
    
    sample = pre_results.loc[uttid].copy()
    dump_dir = dec_model/f"evaluate/{sample.test_split}/results/dump"
    
    uttid2batch = pd.concat([
        pd.DataFrame(np.load(batch), columns=["uttid"])
        .assign(batch_id=int(re.match(r"uttids.(\d+).npy", batch.name).group(1)))
        .assign(sample_id=lambda df: np.arange(len(df)))
        for batch in dump_dir.glob("uttids.*.npy")
    ]).set_index("uttid").sort_index()
    
    sample["dec_raw_pred"] = dec_results.loc[sample.name, "prediction_str"].split(" ")
    sample["cer_dec"], sample["ter_dec"] = float("nan"), dec_results.loc[sample.name, "wer"]
    
    batch_id, sample_id = uttid2batch.loc[sample.name]
    attention_weights = np.load(dump_dir/f"attn_ws.{batch_id:04}.npy")[sample_id]
    plot_attention_predictions(attention_weights[:len(sample.groundtruth.split()), :], sample)
    display(Audio(torch.load(Path(f"~/spchdisk/data/cgn2/{uttid}.pt").expanduser()), rate=16000))
    return attention_weights, sample, dec_results
    
    
def display_audio(uttids, sample_rate=16000):
    for uttid, row in annotations[annotations.index.isin(uttids)].iterrows():
        start, end = (int(t * 16000) for t in (row.start, row.end))
        waveform, _ = torchaudio.load(Path("/users/spraak/spchdata/cgn", row.audio))
        print("{uttid} ({comp}): {text}".format(uttid=uttid, **row))
        waveform = waveform[0, start:end].detach().clone()
        display(Audio(waveform, rate=16000))

        
def format_dataframe_best_metrics(df):

    def highlight_min(data, color='yellow'):
        assert data.ndim == 2
        fmt = data.copy().applymap(lambda _: "")
        return fmt.mask((data.T == data.min(axis=1)).T, f"background-color:{color}")
    
    fmt = df.copy()
    return fmt.groupby(level=1, axis=1).apply(highlight_min)


print(f"Imported objects: {pformat(__all__)}")
# print("Available plotting functions: plot_comparison plot_training plot_eval_results_by_length")