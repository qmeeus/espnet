# coding: utf-8
import os, sys
import warnings
from functools import partial
from operator import itemgetter
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from IPython.display import display


TARGET_FILE = "data/grabo_w2v/encoded_target.csv"
TEXT_FILE = "data/grabo/text"
OUTPUT_DIR = "exp/figures/learning_curve_text"

TARGET_FILE = "/esat/spchdisk/scratch/qmeeus/repos/espnet/egs/patience/sti1/data/encoded_target.csv"
TEXT_FILE = "/esat/spchdisk/scratch/qmeeus/repos/espnet/egs/patience/sti1/data/text"
OUTPUT_DIR = "exp/figures/learning_curve_text_patience"

def plot_learning_curve(estimator,
                        title,
                        X, y,
                        output_file="learning_curve.png",
                        axes=None,
                        ylim=None,
                        cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring="f1_macro",
                       return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.savefig(output_file)
    plt.close('all')
    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


def get_feature_vector(features, method):
    assert method in ("first", "average")
    assert features.ndim == 3
    if method == "first":
        feats = features[:, 0, :]
    else:
        feats = features.mean(1)
    return feats.numpy()


def plot_learning_curve_all(feats, target, target_names, outdir, train_sizes=np.linspace(.1, 1., 5)):
    os.makedirs(outdir, exist_ok=True)

    scores = pd.DataFrame(
        index=target_names,
        columns=pd.MultiIndex.from_product([
            ("train", "test"), ("mean", "std"), train_sizes
        ])
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for i, tn in enumerate(target_names):
            y = target.values[:, i]
            scores.loc[tn] = np.concatenate(
                plot_learning_curve(
                    SVC(C=100., kernel="rbf"), tn, feats, y,
                    f"{outdir}/learning_curve_{tn}.png",
                    n_jobs=-1
                ), axis=0
            )

    averages = scores.xs("mean", axis=1, level=1).mean().unstack(level=0)
    stdevs = ((scores.xs("std", axis=1, level=1) ** 2).mean() ** (1/2)).unstack(level=0)
    for subset, color in zip(("train", "test"), ("r", "g")):
        plt.fill_between(
            train_sizes,
            averages[subset] - stdevs[subset],
            averages[subset] + stdevs[subset],
            alpha=.1, color=color
        )

        plt.plot(train_sizes, averages[subset], "o-", color=color, label=f"{subset} score")

    plt.legend(loc="best")
    plt.savefig(f"{outdir}/scores.png")
    scores.to_csv(f"{outdir}/scores.csv")
    return scores


def read_file(filename):
    uttid = "{3.name}_{1.name}_{0}".format(filename.stem, *filename.parents)
    with open(filename) as f:
        return uttid, f.read().strip()


# Load inputs and outputs
with open(TEXT_FILE) as f:
    texts = (
        pd.DataFrame(
            map(partial(str.split, maxsplit=1),
                map(str.strip, f.readlines())),
            columns=["uttid", "text"]
        ).set_index("uttid")
        .drop_duplicates(subset=["text"])
    )

target = pd.read_csv(TARGET_FILE, index_col="uttid").loc[texts.index]
target_names = list(target.columns)
#target.columns = pd.MultiIndex.from_tuples(map(partial(str.split, sep="_", maxsplit=1), target.columns))

# Load tokenizer and model
model_string = "distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_string)
model = DistilBertModel.from_pretrained(model_string)
for p in model.parameters():
    p.requires_grad = False

# Encode texts
encoded_texts = tokenizer.batch_encode_plus(texts["text"], add_special_tokens=False, padding=True, return_tensors="pt")
features, = model(**encoded_texts)
input_lengths = (encoded_texts["input_ids"] != 0).sum(-1)

# Plot learning curves
for method in ["first", "average"]:
    display(
        plot_learning_curve_all(
            get_feature_vector(features, method),
            target, target_names,
            f"{OUTPUT_DIR}/feats_{method}"
        ).xs("mean", axis=1, level=1)
    )

pd.concat([
    (pd.read_csv(fn, header=[0,1,2], index_col=0)
     .xs(("test", "mean"), axis=1).mean()
     .rename(fn.parent.name.split("_")[1]))
    for fn in Path(OUTPUT_DIR).glob("**/scores.csv")
], axis=1).plot()

plt.savefig(f"{OUTPUT_DIR}/agg_scores.png")

sys.exit(0)
# ****************************************************  STOP ******************************************************

# Load fixed texts
correct_text_loc = Path("/users/spraak/qmeeus/data/grabo/speakers").glob("**/transcription_manual/recording*/*.txt")
with mp.Pool(8) as pool:
    texts.update(
        pd.DataFrame.from_dict(
            dict(pool.map(read_file, correct_text_loc)),
            orient="index"
        ).rename(columns={0: "text"})
    )

invalid_utterances = ["pp2_recording4_Voice_15"]
texts = texts.drop(invalid_utterances)
target = target.drop(invalid_utterances)
encoded_texts = tokenizer.batch_encode_plus(texts["text"], add_special_tokens=False, padding=True, return_tensors="pt")
features, = model(**encoded_texts)
assert (features != features).any(-1).any(-1).sum() == 0, "Features contains nan values"

# Plot learning curves
for method in ["first", "average"]:
    display(
        plot_learning_curve_all(
            get_feature_vector(features, method),
            target, target_names,
            f"exp/figures/learning_curve_text_fixed/feats_{method}"
        ).xs("mean", axis=1, level=1)
    )

pd.concat([
    (pd.read_csv(fn, header=[0,1,2], index_col=0)
     .xs(("test", "mean"), axis=1).mean()
     .rename(fn.parent.name.split("_")[1]))
    for fn in Path("exp/figures/learning_curve_text_fixed").glob("**/scores.csv")
], axis=1).plot()

plt.savefig("exp/figures/learning_curve_text_fixed/agg_scores.png")

