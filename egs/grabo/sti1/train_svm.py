import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import h5py
import argparse
from pathlib import Path
from IPython.display import display
from operator import itemgetter
from functools import partial

from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, classification_report


DATA_ROOT = Path("data/grabo_w2v")
FEATURES = DATA_ROOT/"features.h5"
TARGET = DATA_ROOT/"encoded_target.csv"


class CustomSVC(SVC):

    def score(self, X, y, sample_weight=None, adjusted=False):
        return balanced_accuracy_score(
            y, self.predict(X),
            sample_weight=sample_weight,
            adjusted=adjusted
        )


def get_eval_func(classifier, average='macro'):

    def evaluate(X, y, sample_weight=None):
        yhat = classifier.predict(X)
        acc = balanced_accuracy_score(y, yhat, sample_weight=sample_weight)
        P, R, F, _ = precision_recall_fscore_support(y, yhat, average=average, zero_division=0., sample_weight=sample_weight)
        return acc, P, R, F

    metric_names = ("accuracy", "precision", "recall", "f1_score")
    return metric_names, evaluate

#def hdf5_loader(filename, indices):
#    with h5py.File(filename, "r") as h5f:
#        for key in indices:
#            if key in h5f:
#                yield key, h5f[key][()]
#

def hdf5_loader(filename, indices=None):
    getitem = lambda k: (k, h5f[k][()])
    with h5py.File(filename, "r") as h5f:
        indices = list(filter(lambda k: k in h5f)) if indices else list(h5f.keys())
        yield from map(getitem, indices)


def text2features(texts):
    from transformers import AutoModel, AutoTokenizer
    model_str = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModel.from_pretrained(model_str)
    for param in model.parameters():
        param.requires_grad = False

    encoded_texts = tokenizer.batch_encode_plus(
        texts["text"], add_special_tokens=False, padding=True, return_tensors="pt"
    )

    features, = model(**encoded_texts)
    input_lengths = encoded_texts["attention_mask"].sum(-1)
    return features, input_lengths


def text_loader(filename, indices=None):

    with open(filename, encoding="utf-8") as f:
        texts = pd.DataFrame(
            map(lambda s: s.split(" ", maxsplit=1), map(str.strip, f.readlines())),
            columns=["uttid", "text"]
        ).set_index("uttid").dropna(how='any')

    if indices:
        texts = text.loc[indices]

    indices = list(texts.index)
    features, input_lengths = text2features(texts)

    for i, (j, key) in enumerate(zip(input_lengths, indices)):
        yield key, features[i, :j].numpy()


def load_data(features, targets, indices=None, multiclass=False, combine='first'):
    Loader = hdf5_loader if features.suffix == "h5" else text_loader
    loader = Loader(features, indices)

    if combine == "first":
        combine_func = itemgetter(0)
    elif combine == "average":
        combine_func = partial(np.mean, axis=0)
    else:
        raise NotImplementedError(f"Unknown argument: combine={combine}")

    indices, features = map(list, zip(*map(lambda t: (t[0], combine_func(t[1])), loader)))
    features = np.array(features)
    targets = pd.read_csv(targets, index_col="uttid").loc[indices]
    if multiclass:
        targets.columns = pd.MultiIndex.from_tuples(
            map(lambda c: c.split("_", maxsplit=1), targets.columns))
        targets = targets.groupby(level=0, axis=1).agg(lambda gr: np.argmax(gr.values, axis=1))
    return features, targets.values, targets.index, targets.columns


def has_more_than_one_class(array):
    return len(np.unique(array)) > 1


def train(exp_dir=None,
          features=None,
          targets=None,
          exp_name=None,
          C=10.0,
          kernel='linear',
          class_weight='balanced',
          multiclass=False,
          combine='first',
          random_state=42):

    if multiclass:
        # TODO: #classes per category = #eff class + 1 (account for all zero)
        raise NotImplementedError("multiclass=True not yet available")
    if any(arg is None for arg in (exp_dir, features, targets)):
        raise ValueError("exp_dir, features and targets are required")
    exp_dir = Path(exp_dir)
    (X_train, y_train, train_ids, classes), (X_test, y_test, test_ids, _) = (
        load_data(
            indices=np.loadtxt(exp_dir/f"{subset}.ids", dtype='str'),
            features=features, targets=targets,
            multiclass=multiclass,
            combine=combine
        ) for subset in ("train", "test")
    )

    classifier = CustomSVC(C=C, kernel=kernel, class_weight=class_weight, random_state=random_state)
    train_predictions = pd.DataFrame(index=train_ids, columns=classes)
    test_predictions = pd.DataFrame(index=test_ids, columns=classes)
    for i, target_name in enumerate(classes):
        if not has_more_than_one_class(y_train[:, i]):
            print(f'Class {target_name}: only one class present in dataset')
            train_predictions[target_name] = np.unique(y_train[:, i])[0]
            test_predictions[target_name] = np.unique(y_test[:, i])[0]
            continue

        classifier.fit(X_train, y_train[:, i])
        train_predictions[target_name] = classifier.predict(X_train)
        test_predictions[target_name] = classifier.predict(X_test)

    outdir = exp_dir / exp_name
    os.makedirs(outdir, exist_ok=True)

    with open(outdir/f"train_scores.txt", 'w') as f:
        f.write(classification_report(
            y_train, train_predictions, zero_division=0., target_names=classes))

    with open(outdir/f"test_scores.txt", 'w') as f:
        f.write(classification_report(
            y_test, test_predictions, zero_division=0., target_names=classes))

    train_accuracy, test_accuracy = (
        np.mean([
            balanced_accuracy_score(y_true[:, i], y_pred[target_name])
            for i, target_name in enumerate(classes)
        ]) for y_true, y_pred in ((y_train, train_predictions), (y_test, test_predictions))
    )

    return train_accuracy, test_accuracy

def _train(options):
    return train(**options)

def train_many(tasks, njobs=8, **training_options):
    if len(tasks) == 1 and tasks[0].is_file():
        with open(tasks[0]) as f:
            tasks = list(map(str.strip, f.readlines()))
    elif len(tasks) == 1:
        # Debugging purposes
        train(tasks[0], **training_options)
        return

    print(f"{len(tasks)} tasks in the queue")
    train_args = [
        dict(exp_dir=task, **training_options)
        for task in tasks
    ]

    with mp.Pool(njobs) as pool:
        scores = pool.map(_train, train_args)

    print("\n".join([
        "{}: train_acc={:.3%} test_acc={:.3%}".format(task, *score)
        for task, score in zip(tasks, scores)
    ]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path, nargs="+")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--targets", type=Path, default=TARGET)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--njobs", type=int, default=8)
    parser.add_argument("--C", type=float, default=10.)
    parser.add_argument("--kernel", type=str, choices=['linear', 'rbf', 'sigmoid', 'poly'], default='linear')
    parser.add_argument("--class-weight", type=str, default='balanced')
    parser.add_argument("--multiclass", action="store_true", default=False)
    parser.add_argument("--combine", type=str, choices=["first", "average"], default="first")
    parser.add_argument("--random-state", type=int, default=42)
    train_many(**vars(parser.parse_args()))

