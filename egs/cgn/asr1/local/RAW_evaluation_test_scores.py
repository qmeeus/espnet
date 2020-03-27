# coding: utf-8
import json
def load_logs(path):
    cond = lambda line: any(line.startswith(pfx) for pfx in ("groundtruth", "prediction"))
    with open(path) as f:
        lines = list(filter(cond, f.readlines()))
    outputs = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith("groundtruth"):
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
     return outputs
def load_logs(path):
    cond = lambda line: any(line.startswith(pfx) for pfx in ("groundtruth", "prediction"))
    with open(path) as f:
        lines = list(filter(cond, f.readlines()))
    outputs = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith("groundtruth"):
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
    return outputs
    
path = "exp/decode_pos_data_long_utt/decode.log"
pos = load_logs(path)
pos
def load_logs(path):
    cond = lambda line: any(pfx in line for pfx in ("groundtruth: ", "prediction: "))
    with open(path) as f:
        lines = list(filter(cond, f.readlines()))
    outputs = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith("groundtruth"):
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
    return outputs
    
pos = load_logs(path)
pos
def load_logs(path):
    cond = lambda line: any(pfx in line for pfx in ("groundtruth: ", "prediction: "))
    with open(path) as f:
        lines = f.readlines()
        lines = list(filter(cond, lines))
    outputs = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith("groundtruth"):
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
    return outputs
    
pos = load_logs(path)
def load_logs(path):
    cond = lambda line: any(pfx in line for pfx in ("groundtruth: ", "prediction: "))
    with open(path) as f:
        lines = f.readlines()
        import ipdb; ipdb.set_trace()
        lines = list(filter(cond, lines))
    outputs = []
    for i in range(0, len(lines), 2):
        if lines[i].startswith("groundtruth"):
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
    return outputs
    
pos = load_logs(path)
def load_logs(path):
    cond = lambda line: any(pfx in line for pfx in ("groundtruth: ", "prediction: "))
    with open(path) as f:
        lines = f.readlines()
        lines = list(filter(cond, lines))
    outputs = []
    for i in range(0, len(lines), 2):
        if "groundtruth: " in lines[i]:
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
    return outputs
    
pos = load_logs(path)
def load_logs(path):
    cond = lambda line: any(pfx in line for pfx in ("groundtruth: ", "prediction: "))
    with open(path) as f:
        lines = f.readlines()
        lines = list(filter(cond, lines))
    outputs = []
    for i in range(0, len(lines), 2):
        if "groundtruth: " in lines[i] and i < len(lines) - 1:
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
    return outputs
    
pos = load_logs(path)
pos
def load_logs(path):
    cond = lambda line: any(pfx in line for pfx in ("groundtruth: ", "prediction: "))
    with open(path) as f:
        lines = list(map(str.strip, filter(cond, f.readlines())))
    outputs = []
    for i in range(0, len(lines), 2):
        if "groundtruth: " in lines[i] and i < len(lines) - 1:
            outputs.append({
                "text": lines[i].split("groundtruth: ")[-1],
                "prediction": lines[i+1].split("prediction: ")[-1]
            })
    return outputs
    
pos = load_logs(path)
pos[:10]
def load_logs(path):
    cond = lambda line: any(pfx in line for pfx in ("groundtruth: ", "prediction : "))
    with open(path) as f:
        lines = list(map(str.strip, filter(cond, f.readlines())))
    outputs = []
    while lines:
        cur = lines.pop(0)
        if "groundtruth: " in cur:
            outputs.append({"text": cur.split("groundtruth: ")[-1]})
        else:
            assert "prediction : " in cur, cur
            outputs[-1].update({"prediction": cur.split("prediction : ")[-1]})
    return outputs
    
    
pos = load_logs(path)
pos[-1]
def add_target(outputs, dataset):
    with open(dataset) as jsonfile:
        data = list(json.load(jsonfile)["utts"].items())
    for i, output in enumerate(outputs):
        print(output)
        print(data[i])
        if i == 5:
            break
            
add_target(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json")
def add_target(outputs, dataset):
    with open(dataset) as jsonfile:
        data = list(json.load(jsonfile)["utts"].items())
    for i, ((key, sample), output) in enumerate(zip(data, outputs)):
        print(output)
        print(key)
        print(sample)
        if i == 5:
            break
            
            
add_target(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json")
import pandas as pd
components = pd.read_csv("data/utterances.csv")
components
components = components.set_index("uttid")
components
def add_to_dataframe(outputs, dataset, dataframe):
    dataframe = dataframe.copy()
    with open(dataset) as jsonfile:
        data = list(json.load(jsonfile)["utts"].items())
    cols = ["text", "target", "prediction"]
    dataframe[cols] = None
    for i, ((key, sample), output) in enumerate(zip(data, outputs)):
        text = sample["output"][0]["text"]
        target = sample["output"][0]["token"]
        pred = output["prediction"]
        if text != output["text"]: print(text, output["text"])
        dataframe.loc[key, cols] = text, target, pred
    return dataframe
    
data = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", components)
def add_to_dataframe(outputs, dataset, dataframe):
    dataframe = dataframe.copy()
    with open(dataset) as jsonfile:
        data = list(json.load(jsonfile)["utts"].items())
    cols = ["text", "target", "prediction"]
    for col in cols: dataframe[col] = None
    for i, ((key, sample), output) in enumerate(zip(data, outputs)):
        text = sample["output"][0]["text"]
        target = sample["output"][0]["token"]
        pred = output["prediction"]
        if text != output["text"]: print(text, output["text"])
        dataframe.loc[key, cols] = text, target, pred
    return dataframe
    
filled = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", components)
filled
filled[filled.notnull().any(axis=0)]
filled.notnull(axis=0)
filled.notnull()
filled.notnull().any(axis=0)
filled.notnull().any(axis=1)
filled.notnull().all(axis=1)
filled.notnull().all(axis=1).sum()
filled[filled.notnull().any(axis=1)]
filled[filled.notnull().all(axis=1)]
filled[filled.notnull().all(axis=1)].groupby("comp").count()
filled[filled.notnull().all(axis=1)].groupby("comp").count()
pos = load_logs(path)
filled = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", components)
filled[filled.notnull().all(axis=1)].groupby("comp").count()
wp1000 = load_logs("exp/decode_unigram_1000_data_long_utt_gpu/decode.log")
out_wp1000 = pd.read_csv("data/utterances.csv")
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", out_wp1000)
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_unigram_1000.lg.150+.json", out_wp1000)
def summary_by_component(df):
    return df[df.notnull().all(axis=1)].groupby("comp").count()
    
summary_by_component(out_wp1000)
out_wp1000
out_wp1000.head()
out_wp1000.tail()
out_wp1000 = pd.read_csv("data/utterances.csv")
wp1000 = load_logs("exp/decode_unigram_1000_data_long_utt_gpu/decode.log")
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_unigram_1000.lg.150+.json", out_wp1000)
wp1000 = load_logs("exp/decode_unigram_1000_data_long_utt_gpu/decode.log")
len(wp1000)
type(wp1000)
wp1000[0]
def add_to_dataframe(outputs, dataset, dataframe):
    dataframe = dataframe.copy()
    with open(dataset) as jsonfile:
        data = list(json.load(jsonfile)["utts"].items())
    cols = ["text", "target", "prediction"]
    for col in cols: dataframe[col] = None
    keys, values = [], []
    for i, ((key, sample), output) in enumerate(zip(data, outputs)):
        text = sample["output"][0]["text"]
        target = sample["output"][0]["token"]
        pred = output["prediction"]
        if text != output["text"]: print(text, output["text"])
        keys.append(key)
        values.append((text, target, pred))
    dataframe.loc[keys, cols] = values
    return dataframe
    
    
wp1000 = load_logs("exp/decode_unigram_1000_data_long_utt_gpu/decode.log")
out_wp1000 = pd.read_csv("data/utterances.csv")
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_unigram_1000.lg.150+.json", out_wp1000)
out_wp1000
out_wp1000 = pd.read_csv("data/utterances.csv").set_index("uttid")
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_unigram_1000.lg.150+.json", out_wp1000)
out_wp1000
summary_by_component(out_wp1000)
out_wp1000[out_wp1000.notnull().all()]
out_wp1000[out_wp1000.notnull().all(axis=1)]
from editdistance import eval as editdistance
def wer(pred, target):
    return editdistance(pred.split(), target.split()) / len(target.split())
    
out_wp1000[out_wp1000.notnull().all(axis=1)].iloc[0,-1]
PHR = out_wp1000[out_wp1000.notnull().all(axis=1)].iloc[0,-1][0]
PHR
out_wp1000["wer"] = None
out_wp1000["cer"] = None
out_wp1000.loc[out_wp1000.notnull().all(axis=1), "wer"] = out_wp1000.loc[out_wp1000.notnull().all(axis=1), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]))
pd.DataFrame.where?
out_wp1000.loc[out_wp1000.notnull().all(axis=1), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]))
out_wp1000.loc[out_wp1000.notnull().all(axis=1), ["prediction", "target"]]
out_wp1000.loc[out_wp1000.notnull().any(axis=1), ["prediction", "target"]]
out_wp1000.loc[out_wp1000.notnull().all(axis=1), ["prediction", "target"]]
summary_by_component(out_wp1000)
def summary_by_component(df):
    return df[df["target"].notnull().all(axis=1)].groupby("comp").count()
    
    
summary_by_component(out_wp1000)
def summary_by_component(df):
    return df[df["target"].notnull()].groupby("comp").count()
    
    
    
summary_by_component(out_wp1000)
out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]]
out_wp1000.loc[out_wp1000.notnull().all(axis=1), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]))
out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]))
out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]), axis=1)
out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].assign(wer=lambda row: wer(row["prediction"], row["target"]), axis=1)
out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]), axis=1).rename("wer")
out_wp1000.update(out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]), axis=1).rename("wer"))
out_wp1000[out_wp1000["target".notnull()]].groupby("comp").mean()
out_wp1000[out_wp1000["target"].notnull()].groupby("comp").mean()
out_wp1000.loc[out_wp1000["target"].notnull(), "wer"].groupby("comp").mean()
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].groupby("comp").mean()
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]]
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].groupby("comp").mean()
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].groupby("comp").sum()
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].groupby("comp").mean()
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].groupby("comp").agg({"wer": "mean"})
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].groupby("comp")
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]]
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].summary()
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].describe()
out_wp1000.dtypes
out_wp1000["wer"] = pd.to_numeric(out_wp1000["wer"])
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer"]].groupby("comp").mean()
def cer(pred, target):
    pred, target = (val.replace(" ", "").replace(PHR, "") for val in (pred, target))
    return editdistance(pred, target) / len(target)
    
    
pred, target = out_wp1000.loc[out_wp1000.target.notnull(), ["prediction", "target"]].iloc[0]
pred, target
def cer(pred, target):
    pred, target = (val.replace(" ", "").replace(PHR, "").replace("<eos>", "").strip() for val in (pred, target))
    return editdistance(pred, target) / len(target)
    
    
cer(pred, target)
pred[:20]
cer(pred[:20], target[:20])
pred[:30]
pred[:40]
pred[:50]
pred[:60]
target[:60]
out_wp1000.update(out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: cer(row["prediction"], row["target"]), axis=1).rename("cer"))
out_wp1000["cer"] = pd.to_numeric(out_wp1000["cer"])
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer", "cer"]].groupby("comp").mean()
wp1000 = load_logs("exp/decode_unigram_1000_data_long_utt_gpu/decode.log")
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_unigram_1000.lg.150+.json", out_wp1000)
out_wp1000.update(out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: cer(row["prediction"], row["target"]), axis=1).rename("cer"))
out_wp1000.update(out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]), axis=1).rename("wer"))
out_wp1000.loc[out_wp1000["target"].notnull(), ["comp", "wer", "cer"]].groupby("comp").mean()
def add_to_dataframe(outputs, dataset, dataframe, prefix):
    dataframe = dataframe.copy()
    with open(dataset) as jsonfile:
        data = list(json.load(jsonfile)["utts"].items())
    cols = ["text", prefix + "target", prefix + "prediction"]
    for col in cols: 
        if col not in dataframe:
            dataframe[col] = None
    keys, values = [], []
    for i, ((key, sample), output) in enumerate(zip(data, outputs)):
        text = sample["output"][0]["text"]
        target = sample["output"][0]["token"]
        pred = output["prediction"]
        if text != output["text"]: print(text, output["text"])
        keys.append(key)
        values.append((text, target, pred))
    dataframe.loc[keys, cols] = values
    return dataframe
    
    
    
out_wp1000 = out_wp1000.rename(columns={"prediction": "wp_prediction", "target": "wp_target"})
def add_to_dataframe(outputs, dataset, dataframe, prefix="wp_"):
    dataframe = dataframe.copy()
    with open(dataset) as jsonfile:
        data = list(json.load(jsonfile)["utts"].items())
    cols = ["text", prefix + "target", prefix + "prediction"]
    for col in cols: 
        if col not in dataframe:
            dataframe[col] = None
    keys, values = [], []
    for i, ((key, sample), output) in enumerate(zip(data, outputs)):
        text = sample["output"][0]["text"]
        target = sample["output"][0]["token"]
        pred = output["prediction"]
        if text != output["text"]: print(text, output["text"])
        keys.append(key)
        values.append((text, target, pred))
    dataframe.loc[keys, cols] = values
    return dataframe
        
out_wp1000
pos = load_logs("exp/decode_pos_data_long_utt_fasso/decode.log")
pos[:10]
out_pos = pd.read_csv("data/utterances.csv").set_index("uttid")
out_pos = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos.lg.150+.json", out_pos)
out_pos = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", out_pos)
summary_by_component(out_pos)
def summary_by_component(df):
    return df[df["text"].notnull()].groupby("comp").count()
    
summary_by_component(out_pos)
out_pod
out_pos
out_pos[out_pos.text.notnull()]
out_pos = out_pos.rename(columns={"wp_target": "pos_target", "wp_prediction": "pos_prediction"})
out_pos[out_pos.text.notnull()]
import re
pos_target_classes = pos_target.text.str.replace(re.compile("\(.*?\)"), "")
pos_target_classes = out_pos.pos_target.text.str.replace(re.compile("\(.*?\)"), "")
pos_target_classes = out_pos.pos_target.str.replace(re.compile("\(.*?\)"), "")
pos_target_classes
pos_target_classes[pos_target_classes.notnull()]
pos_pred_classes = out_pos.pos_prediction.str.replace(re.compile("\(.*?\)"), "")
class_wer = pd.concat([pos_pred_classes, pos_target_classes], axis=1).apply(lambda row: wer(row["pos_prediction"], row["pos_target"]), axis=1)
class_wer = pd.concat([pos_pred_classes, pos_target_classes], axis=1).dropna(how='any', axis=0).apply(lambda row: wer(row["pos_prediction"], row["pos_target"]), axis=1)
class_wer
out_pos["class_error_rate"] = np.NaN
out_pos["class_error_rate"] = float("nan")
class_wer = class_wer.rename("class_error_rate")
out_pos.update(class_wer)
out_pos
out_pos.dropna(how="any", axis=1)
out_pos.dropna(how="any", axis=0)
wer("hello world", "world hello")
wer("hello world this is quentin", "world hello this is quentin")
editdistance(["hello", "world"], ["world", "hello"])
editdistance("tu", "ut")
out_pos[out_pos.text.notnull()].apply(lambda row: wer(row["prediction"], row["target"]), axis=1).rename("error_rate")
out_pos[out_pos.text.notnull()].apply(lambda row: wer(row["pos_prediction"], row["pos_target"]), axis=1).rename("error_rate")
out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: wer(row["pos_prediction"], row["pos_target"]), axis=1).rename("error_rate"))
out_pos = out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: wer(row["pos_prediction"], row["pos_target"]), axis=1).rename("error_rate"))
out_pos.dropna(how="any", axis=1)
out_pos.dropna(how="any", axis=0)
out_pos[["comp", "name", "class_error_rate", "error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "error_rate": "mean"})
out_pos = out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: wer(row["pos_prediction"], row["pos_target"].replace("()", "")), axis=1).rename("error_rate"))
out_pos = out_pos.drop("error_rate", axis=1)
out_pos = out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: wer(row["pos_prediction"], row["pos_target"].replace("()", "")), axis=1).rename("error_rate"))
out_pos[["comp", "name", "class_error_rate", "error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "error_rate": "mean"})
def tag_error_rate(pred, target):
    target = target.replace("()", "")
    pred, target = (s.split(" ") for s in (pred, target))
    return distance
def tag_error_rate(pred, target):
    target = target.replace("()", "")
    
    def parse_tag(tag):
        if "(" in tag:
            head, tail = tag.split("(")
            tail = tail[:-1]
        else:
            head, tail = tag, None
        return head, tail
    pred, target = (s.split(" ") for s in (pred, target))
def tag_error_rate(pred, target):
    target = target.replace("()", "")
    
    def parse_tag(tag):
        if "(" in tag:
            head, tail = tag.split("(")
            tail = tail[:-1]
        else:
            head, tail = tag, None
        return head, tail
    pred, target = (s.split(" ") for s in (pred, target))
def tag_error_rate(pred, target):
    target = target.replace("()", "")
    
    def parse_tag(tag):
        if "(" in tag:
            head, tail = tag.split("(")
            tail = ",".split(tail[:-1])
        else:
            head, tail = tag, None
        return head, tail
        
    pred, target = (list(map(parse_tag, s.split(" "))) for s in (pred, target))
    length = len(target)
    (pred_heads, pred_tails), (target_heads, target_tails) = (map(list, zip(*l)) for l in (pred, target)) 
    class_error_rate = editdistance(pred_heads, target_heads) / length
    error_rate = np.mean([
        1 if phead != thead else (
            0 if not ttail else editdistance(ptail, ttail) / len(ttail)
        ) for (phead, ptail), (thead, ttail) in zip(pred, target)
    ])
    return class_error_rate, error_rate
    
    
    
out_pos.dropna(how="any", axis=0).iloc[0]
target, prediction = out_pos.dropna(how="any", axis=0).iloc[0][["pos_target", "pos_prediction"]]
tag_error_rate(prediction, target)
import numpy as np
tag_error_rate(prediction, target)
def tag_error_rate(pred, target):
    target = target.replace("()", "")
    
    def parse_tag(tag):
        if "(" in tag:
            head, tail = tag.split("(")
            tail = ",".split(tail[:-1])
        else:
            head, tail = tag, None
        return head, tail
        
    pred, target = (list(map(parse_tag, s.split(" "))) for s in (pred, target))
    length = len(target)
    (pred_heads, pred_tails), (target_heads, target_tails) = (map(list, zip(*l)) for l in (pred, target)) 
    class_error_rate = editdistance(pred_heads, target_heads) / length
    error_rate = np.mean([
        1 if phead != thead else (
            0 if ttail is None else editdistance(ptail, ttail) / len(ttail)
        ) for (phead, ptail), (thead, ttail) in zip(pred, target)
    ])
    return class_error_rate, error_rate
    
    
    
tag_error_rate(prediction, target)
prediction, target
def tag_error_rate(pred, target):
    target = target.replace("()", "")
    
    def parse_tag(tag):
        if "(" in tag:
            head, tail = tag.split("(")
            tail = ",".split(tail[:-1])
        else:
            head, tail = tag, None
        return head, tail
        
    pred, target = (list(map(parse_tag, s.split(" "))) for s in (pred, target))
    length = len(target)
    (pred_heads, pred_tails), (target_heads, target_tails) = (map(list, zip(*l)) for l in (pred, target)) 
    class_error_rate = editdistance(pred_heads, target_heads) / length
    error_rate = np.mean([
        1 if phead != thead else (
            0 if ttail is None else editdistance(ptail, ttail) / len(ttail)
        ) for (phead, ptail), (thead, ttail) in zip(pred, target)
    ])
    return class_error_rate, error_rate
    
    
    
out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1)
list(zip(*out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1)))
out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).map(pd.Series)
out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series)
out_pos
out_pos = out_pos.drop(["class_error_rate", "error_rate"], axis=1)
out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename({0: "class_error_rate", 1: "tag_error_rate"}))
out_pos = out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename({0: "class_error_rate", 1: "tag_error_rate"}))
out_pos
out_pot = out_pos.drop([0,1], axis=1)
out_pos = out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename(columns={0: "class_error_rate", 1: "tag_error_rate"}))
out_pos
out_pos = out_pos.drop([0,1], axis=1)
out_pos
out_pos.dropna(how="all", axis=0)
out_pos.dropna(how="any", axis=0)
out_pos[["comp", "name", "class_error_rate", "error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "error_rate": "mean"})
out_pos[["comp", "name", "class_error_rate", "error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "tag_error_rate": "mean"})
out_pos[["comp", "name", "class_error_rate", "tag_error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "tag_error_rate": "mean"})
pos = load_logs("exp/decode_pos_data_long_utt_fasso/decode.log")
out_pos = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", out_pos)
out_pos
out_pos = out_pos.drop(["wp_target", "wp_prediction"])
out_pos = out_pos.drop(["wp_target", "wp_prediction"], axis=1)
out_pos
out_pos = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", out_pos, "pos_")
out_pos.text.notnull().sum()
out_pos = out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename(columns={0: "class_error_rate", 1: "tag_error_rate"}))
out_pos = out_pos.drop(["class_error_rate", "tag_error_rate"], axis=1)
out_pos = out_pos.join(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename(columns={0: "class_error_rate", 1: "tag_error_rate"}))
out_pos[["comp", "name", "class_error_rate", "tag_error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "tag_error_rate": "mean"})
wp1000 = load_logs("exp/decode_unigram_1000_data_long_utt_gpu/decode.log")
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_unigram_1000.lg.150+.json", out_wp1000)
wp1000
out_wp1000.update(out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]), axis=1).rename("wer"))
out_wp1000.update(out_wp1000.loc[out_wp1000["target"].notnull(), ["wp_prediction", "wp_target"]].apply(lambda row: wer(row["wp_prediction"], row["wp_target"]), axis=1).rename("wer"))
out_wp1000.update(out_wp1000.loc[out_wp1000.text.notnull(), ["wp_prediction", "wp_target"]].apply(lambda row: wer(row["wp_prediction"], row["wp_target"]), axis=1).rename("wer"))
out_wp1000.update(out_wp1000.loc[out_wp1000["wp_target"].notnull(), ["wp_prediction", "wp_target"]].apply(lambda row: cer(row["wp_prediction"], row["wp_target"]), axis=1).rename("cer"))
out_wp1000[["comp", "name", "wer", "cer"]].groupby("comp").agg({"name": "count", "wer": "mean", "cer": "mean"})
out_wp1000[["comp", "name", "wer", "cer"]].groupby("comp").agg({"name": "count", "wer": "mean", "cer": "mean"}).rename(columns={"name": "count"})
out_pos[["comp", "name", "class_error_rate", "tag_error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "tag_error_rate": "mean"})
pos = load_logs("exp/decode_pos_data_long_utt_fasso/decode.log")
out_pos = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", out_pos, "pos_")
out_pos.update(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename(columns={0: "class_error_rate", 1: "tag_error_rate"}))
out_pos[["comp", "name", "class_error_rate", "tag_error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "tag_error_rate": "mean"})
scores_pos = out_pos[["comp", "name", "class_error_rate", "tag_error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "tag_error_rate": "mean"})
scores_pos
scores_pos.class_error_rate * scores_pos.name / scores_pos.name.sum()
(scores_pos.class_error_rate * scores_pos.name / scores_pos.name.sum()).sum()
scores_wp1000 = out_wp1000[["comp", "name", "wer", "cer"]].groupby("comp").agg({"name": "count", "wer": "mean", "cer": "mean"}).rename(columns={"name": "count"})
(scores_wp1000.cer * scores_wp1000["count"] / scores_wp1000["count"].sum()).sum()
pos = load_logs("exp/decode_pos_data_long_utt_fasso/decode.log")
out_pos = add_to_dataframe(pos, "dump/test_m/deltafalse/data_pos_300.lg.150+.json", out_pos, "pos_")
out_pos.update(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename(columns={0: "class_error_rate", 1: "tag_error_rate"}))
scores_pos = out_pos[["comp", "name", "class_error_rate", "tag_error_rate"]].groupby("comp").agg({"name": "count", "class_error_rate": "mean", "tag_error_rate": "mean"})
scores_pos
wp1000 = load_logs("exp/decode_unigram_1000_data_long_utt_gpu/decode.log")
out_wp1000 = add_to_dataframe(wp1000, "dump/test_m/deltafalse/data_unigram_1000.lg.150+.json", out_wp1000)
out_wp1000.update(out_wp1000.loc[out_wp1000["target"].notnull(), ["prediction", "target"]].apply(lambda row: wer(row["prediction"], row["target"]), axis=1).rename("wer"))
out_pos.update(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename(columns={0: "class_error_rate", 1: "tag_error_rate"}))
out_pos.update(out_pos[out_pos.text.notnull()].apply(lambda row: tag_error_rate(row["pos_prediction"], row["pos_target"]), axis=1).apply(pd.Series).rename(columns={0: "class_error_rate", 1: "tag_error_rate"}))
out_wp1000.update(out_wp1000.loc[out_wp1000["wp_target"].notnull(), ["wp_prediction", "wp_target"]].apply(lambda row: cer(row["wp_prediction"], row["wp_target"]), axis=1).rename("cer"))
out_wp1000.update(out_wp1000.loc[out_wp1000.text.notnull(), ["wp_prediction", "wp_target"]].apply(lambda row: wer(row["wp_prediction"], row["wp_target"]), axis=1).rename("wer"))
scores_wp1000 = out_wp1000[["comp", "name", "wer", "cer"]].groupby("comp").agg({"name": "count", "wer": "mean", "cer": "mean"}).rename(columns={"name": "count"})
scores_wp1000
%save -r local/RAW_evaluation_test_scores.py 1-269
