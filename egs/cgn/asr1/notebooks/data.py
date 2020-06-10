import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display
from pprint import pformat

pd.set_option("display.float_format", "{:.4f}".format)

__all__ = [
    "re",
    "json",
    "pd",
    "np",
    "Path",
    "display",
    "logfile_register",
    "datasets",
    "criterion",
    "ascending",
    "option_names",
    "all_options",
    "available_models",
    "update_logfile_register",
    "parse_filenames",
    "load_logs",
    "filter_dataframe",
    "load_evaluation_results", 
    "get_n_bests", 
    "eval_result_summary_by_length"

]


logfile_register = "../exp/logfiles.txt"
datasets = ["o", "ok", "mono", "all", "shuffle"]
criterion, ascending = "validation/main/cer_ctc", True
option_names = [
    "target", "dataset", "model_type", "elayers", "dlayers", "units", 
    "adim", "ahead", "alpha", "dropout_enc", "dropout_dec", "weight_decay"
]

all_options = [*option_names, "path", "exp_name", "model_name", "task", "logfile", "eval_dir"]


def update_logfile_register():
    logfiles = np.array(list(map(str, Path("../exp").glob("train_*/**/results/log"))))
    np.savetxt(logfile_register, logfiles, fmt="%s")


def parse_filenames(update=False):
    
    models = pd.read_csv(logfile_register, names=["logfile"], squeeze=True).map(Path)
        
    def create_rgx(is_lstm=True):

        # ../exp/train_unigram_1000_mono/lstmp_6_1_1024_a.1_do.1.1_wd5/train
        # ../exp/train_unigram_1000_mono/lstmp_6_1_1024_att_do.05.05_wd5/train
        # ../exp/train_unigram_1000_mono/lstmp_6_1_1024_ctc_do.1.1_wd5/dec-only
        # ../exp/train_unigram_1000_mono/lstmp_6_1_1024_a.1_do.1/retrain
        # ../exp/train_unigram_1000_mono/lstmp_6_1_1024_a.1_do.1.1/train
        # ../exp/train_unigram_1000_ok/transformer_12_6_2048_256_4_a.3_do.1/train
        
        model_config = "\d+" if is_lstm else "\d+_\d+_\d+"
        return re.compile(
            r"(?P<path>\.\.\/exp\/(?P<exp_name>train_(?P<target>\w+)_(?P<dataset>[a-z]+))\/"
            r"(?P<model_name>(?P<model_type>[a-z]+)_(?P<elayers>\d+)_(?P<dlayers>\d+)_"
            r"(?P<units>\d+)_(?:(?P<adim>\d+)_(?P<ahead>\d+)_)?"
            r"(?P<alpha>(?:a\.\d+)|(?:ctc)|(?:att))"
            r"(?:_do(?P<dropout_enc>\.\d+)(?:(?P<dropout_dec>\.\d+))?)?(?:_wd(?P<weight_decay>\d+))?(?:\w+)?)\/"
            r"(?P<task>[\w-]+))"
        )
    
    models = pd.DataFrame(index=models.index, columns=all_options).assign(logfile=models)
    models.update(models["logfile"].map(str).str.extract(create_rgx()))

    def convert(col, typ, default=0):
        def _convert(df):
            fillvalue = df[default] if type(default) is str else default
            return pd.to_numeric(df[col].fillna(fillvalue)).astype(typ)
        return _convert

    def convert_alpha(s):   
        if not s: return
        elif s == "att": return 0.
        elif s == "ctc": return 1.
        else: return float(s.replace("a", ""))

    def find_eval_dir(path):
        path = Path(path)
        for potential in (path, path.parent):
            if (potential / "evaluate").exists():
                return potential / "evaluate"

    transforms = {
        'dataset': lambda df: pd.Categorical(df["dataset"], datasets),
        'path': lambda df: df["path"].map(Path, na_action="ignore"),
        'logfile': lambda df: df["logfile"].map(Path, na_action="ignore"),
        'alpha': lambda df: df["alpha"].map(convert_alpha, na_action="ignore"),
        'eval_dir': lambda df: df["path"].map(find_eval_dir, na_action="ignore"),
        **{col: convert(col, typ, d) for col, typ, d in zip(
            ["units", "adim", "ahead", "elayers", "dlayers", "dropout_enc", "dropout_dec", "weight_decay"],
            [int, int, int, int, int, float, float, int],
            [0, "units", 1, 0, 0, 0, 0, 0]
        )}
    }
    
    models.update(
        models.loc[models["dataset"] == "curriculum", "logfile"]
        .map(lambda s: s.split("/")[-3])
        .rename("dataset")
    )
    
    models = (
        models.assign(**transforms)
        .sort_values(option_names)
        .reset_index(drop=True)
        [all_options]
        .copy()
    )

    return models


def load_logs(models):

    column_mapping = {
        "decoder/loss": "main/loss"
    }

    return pd.concat([
        pd.read_json(logfile).assign(model_id=i).rename(columns=column_mapping)
        for i, logfile in models["logfile"].iteritems()
    ], axis=0).join(models, on="model_id")


def filter_dataframe(df, **filters):
    for key, value in filters.items():
        df = df.loc[df[key] == value]
    return df.copy()


def get_n_bests(results, selection="best_iter", criterion=criterion, ascending=ascending, n=5):
    results = results.copy()
    
    group_keys = ["exp_name", "model_name"]
    
    if selection == "best_epoch":
        results = results.groupby(group_keys + ["epoch"], as_index=False).mean()
    
    return (
        results
        .sort_values(criterion, ascending=ascending)
        .groupby(group_keys).head(1)
        .sort_values(criterion, ascending=ascending)
        .head(n)
        .set_index(group_keys)
    )


def load_evaluation_results(model_dir, test_sets=list("abfghijklmno"), verbose=False):

    def _load_evaluation_results(model_dir, test_sets, template, orient="index"):

        def load_json(test_set):
            jsonfile = template.format(model_dir=model_dir, test=test_set)
            if not Path(jsonfile).exists():
                return {}
            with open(jsonfile) as f:
                return list(json.load(f).values())[0]

        return pd.concat(filter(len, [
            pd.DataFrame.from_dict(load_json(test_set), orient=orient).assign(test_split=test_set)
            for test_set in test_sets
        ]), axis=0)

    if "curriculum" in str(model_dir):
        raise NotImplementedError

    templates = [
        "{model_dir}/evaluate/{test}/results.json",
        "{model_dir}/evaluate/{test}/results/results.json",
        "{model_dir}/evaluate/results.{test}.json"
    ]

    for i, (template, orient) in enumerate(zip(templates, ["index", "index", "columns"])):
        try:
            if verbose: print(template)
            results = _load_evaluation_results(model_dir, test_sets, template, orient)
            return results
        except Exception as err:
            if verbose: print(f"Tentative #{i} failed: {err}")

    raise TypeError(model_dir)


def eval_result_summary_by_length(eval_results, q=10):
    return (
        eval_results.groupby([
            "test_split", 
            pd.qcut(eval_results["groundtruth"].str.split(" ").map(len), q)
        ]).describe()
    )


# print("Loading all available models...")
if not Path(logfile_register).exists():
    update_logfile_register()
    
available_models = parse_filenames()
# print(f"{len(available_models)} models loaded:")
# display(available_models.groupby(["exp_name"]).agg({"model_name": "count"}).sort_values("model_name", ascending=False))
# display(available_models.apply(pd.Series.nunique, axis=0).sort_values(ascending=False).rename("count").to_frame())

print(f"Imported objects: {pformat(__all__)}")
# print("Available functions: parse_filenames load_logs filter_dataframe load_evaluation_results get_n_bests eval_result_summary_by_length")
# print("Variables:")
# print(f"\tdatasets = {datasets}")
# print(f"\toption_names = {option_names}")
# print(f"\tall_options = {all_options}")
# print(f"\tcriterion = {criterion} ascending = {ascending}\n")