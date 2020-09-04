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


def load_data(indices, features, targets, multiclass=False, combine='first'):
    if combine == "first":
        combine_func = itemgetter(0)
    elif combine == "mean":
        combine_func = partial(np.mean, axis=0)
    else:
        raise NotImplementedError(f"Unknown argument: combine={combine}")

    with h5py.File(features, "r") as h5f:
        features = np.array([combine_func(h5f[key][()]) for key in indices])
    targets = pd.read_csv(targets, index_col="uttid").loc[indices]
    if multiclass:
        targets.columns = pd.MultiIndex.from_tuples(map(lambda c: c.split("_", maxsplit=1), targets.columns))
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
            np.loadtxt(exp_dir/f"{subset}.ids", dtype='str'),
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
            test_predictions[target_name] = np.unique(y_train[:, i])[0]
            continue

        classifier.fit(X_train, y_train[:, i])
        train_predictions[target_name] = classifier.predict(X_train)
        test_predictions[target_name] = classifier.predict(X_test)

    suffix = "" if exp_name is None else f"_{exp_name}"
    with open(exp_dir/f"train_scores{suffix}.txt", 'w') as f:
        f.write(classification_report(y_train, train_predictions, zero_division=0., target_names=classes))

    with open(exp_dir/f"test_scores{suffix}.txt", 'w') as f:
        f.write(classification_report(y_test, test_predictions, zero_division=0., target_names=classes))

    test_predictions.to_pickle(exp_dir/f"test_predictions{suffix}.pkl")

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
    parser.add_argument("--features", type=Path, default="data/grabo_w2v/features.h5")
    parser.add_argument("--targets", type=Path, default="data/grabo_w2v/encoded_target.h5")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--njobs", type=int, default=8)
    parser.add_argument("--C", type=float, default=10.)
    parser.add_argument("--kernel", type=str, choices=['linear', 'rbf', 'sigmoid', 'poly'], default='linear')
    parser.add_argument("--class_weight", type=str, default='balanced')
    parser.add_argument("--multiclass", action="store_true", default=False)
    parser.add_argument("--combine", type=str, choices=['first', 'mean'], default='first')
    parser.add_argument("--random_state", type=int, default=42)
    train_many(**vars(parser.parse_args()))

