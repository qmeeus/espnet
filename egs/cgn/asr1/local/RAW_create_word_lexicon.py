# coding: utf-8
import json
with open("data.json") as f:
    metadata = json.load(f)
    
metadata
metadata = metadata["utts"]
metadata
metadata.keys()
ids, metadata = sorted(metadata.items(), key=lambda t: t[0])
ids, metadata = zip(*sorted(metadata.items(), key=lambda t: t[0]))
ids
type(metadata)
len(metadata)
metadata = list(metadata)
metadata[0]
outputs = [d["output"][0] for d in metadata]
outputs[0]
from collections import Counter
counter = Counter()
[counter.add([word for word in output["text"].split()]) for output in outputs]
counter.update?
[counter.update([word for word in output["text"].split()]) for output in outputs]
counter
len(counter)
counter.items()
lexicon = []
lexicon = ["<unk>"]
sorted(counter)
lexicon = sorted(counter)
lexicon.insert(0, lexicon.pop(lexicon.index("<unk>")))
lexicon[:10]
lexicon[:100]
len(lexicon)
pwd
ls
import os
# with open("../../../data/lang_1word/train_s__units.txt", "w") as f:
os.makedirs("../../../data/lang_1word")
with open("../../../data/lang_1word/train_s__units.txt", "w") as f:
    f.write("\n".join((f"{word} {i}" for i, word in enumerate(lexicon, 1))))
    
outputs
outputs[0]
outputs2 = []
for output in outputs:
    words = output["text"].split()
    output2 = {
        "name": "target2",
        "shape": [len(words), len(lexicon)],
        "text": output["text"],
        "tokenid": " ".join((word2token[word] for word in words))
    }
    outputs2.append(output2)
    
word2token = dict([(word, i) for i, word in enumerate(lexicon, 1)])
word2token
for output in outputs:
    words = output["text"].split()
    output2 = {
        "name": "target2",
        "shape": [len(words), len(lexicon)],
        "text": output["text"],
        "tokenid": " ".join((word2token[word] for word in words))
    }
    outputs2.append(output2)
    
for output in outputs:
    words = output["text"].split()
    output2 = {
        "name": "target2",
        "shape": [len(words), len(lexicon)],
        "text": output["text"],
        "tokenid": " ".join((str(word2token[word]) for word in words))
    }
    outputs2.append(output2)
    
outputs2[0]
outputs[:5]
outputs2[:5]
len(metadata)
type(metadata)
metadata[0]
for i in range(len(metadata)): metadata[i]["output"] = [outputs2[i]]
metadata = {"utts": metadata}
metadata.keys()
ls
cd ..
ls
cd ..
mkdir train_s_words_
cd train_s_words_/
with open("data.json", "w") as f:
    
    json.dump(metadata, f)
    
cd ../
ls
cd dev_s/
ls
cd deltafalse/
ls
with open("data.json") as f:
    metadata = json.load(f)
    
metadata = metadata["utts"]
cd ../../train_s_/
with open("data.json") as f:
    metadata = json.load(f)
    
cd deltafalse/
with open("data.json") as f:
    metadata = json.load(f)
    
metadata = metadata["utts"]
ids, metadata = zip(*sorted(metadata.items(), key=lambda t: t[0]))
outputs = [d["output"][0] for d in metadata]
outputs2 = []
for output in outputs:
    words = output["text"].split()
    output2 = {
        "name": "target2",
        "shape": [len(words), len(lexicon)],
        "text": output["text"],
        "tokenid": " ".join((str(word2token[word]) for word in words))
    }
    outputs2.append(output2)
    
for i in range(len(metadata)): metadata[i]["output"] = [outputs2[i]]
metadata[0]
metadata = dict(zip(ids, metadata))
metadata["V40194-fv801122.6"]
metadata = {"utts": metadata}
cd ../../
cd train_s_words_/
with open("data.json", "w") as f:
    
    json.dump(metadata, f)
    
cd ../dev_s/deltafalse/
ls
with open("data.json") as f:
    metadata = json.load(f)
    
metadata = metadata["utts"]
ids, metadata = zip(*sorted(metadata.items(), key=lambda t: t[0]))
outputs = [d["output"][0] for d in metadata]
outputs2 = []
from collections import defaultdict
defaultdict?
w2t = defaultdict("<unk>")
for output in outputs:
    words = output["text"].split()
    output2 = {
        "name": "target2",
        "shape": [len(words), len(lexicon)],
        "text": output["text"],
        "tokenid": " ".join((str(word2token.get(word, 1)) for word in words))
    }
    outputs2.append(output2)
    
outputs2[0]
for i in range(len(metadata)): metadata[i]["output"] = [outputs2[i]]
metadata = dict(zip(ids, metadata))
metadata = {"utts": metadata}
ls
cd ../
cd ..
ls
mkdir dev_s_words
cd dev_s_words/
with open("data.json", "w") as f:
    
    json.dump(metadata, f)
    
ls
cd ../
cd ../
ls
cd local/
ls
%save -r RAW_create_word_lexicon.py 1-115
