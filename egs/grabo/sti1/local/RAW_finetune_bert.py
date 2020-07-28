# coding: utf-8
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


EPOCHS = 1000
BATCH_SIZE = 64
ES_MIN_EPOCHS = 1000
ES_PATIENCE = 10
device = torch.device("cuda:1")

def iter_batches(subset, bs=32, shuffle=True, device='cuda', padding_idx=0):
    subset = subset.sample(frac=1.).copy()
    start = 0
    while start < len(subset):
        end = min(start + bs, len(subset))
        Xb, yb = subset["X"].iloc[start:end], subset["y"].iloc[start:end].tolist()
        Xb = pad_sequence(Xb, padding_idx)
        att_mask = get_attention_masks(Xb, padding_idx)
        Xb = torch.tensor(Xb).long().to(device=device)
        yb = torch.tensor(yb).long().to(device)
        att_mask = torch.tensor(att_mask).bool().to(device)
        yield Xb, yb, att_mask
        start = end

def pad_sequence(seq, pad_value=0):
    maxlen = max(map(len, seq))
    bs = len(seq)
    array = np.ones((bs, maxlen)) * pad_value
    for i, s in enumerate(seq):
        array[i, :len(s)] = s
    return array

def get_attention_masks(padded_encodings, padding_idx=0):
        attention_masks = []
        for encoding in padded_encodings:
            attention_mask = [int(token_id != padding_idx) for token_id in encoding]
            attention_masks.append(attention_mask)
        return attention_masks


data = pd.read_csv("data/grabo/target.csv", usecols=["text", "action_string"]).drop_duplicates(ignore_index=True)
X, y = data.text, data.action_string

tokenizer = BertTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
Xenc = X.map(tokenizer.encode)

y_encoder = LabelEncoder().fit(y)
yenc = y_encoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(Xenc, yenc)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.8)
train = pd.DataFrame().assign(X=X_train, y=y_train)
valid = pd.DataFrame().assign(X=X_val, y=y_val)
test = pd.DataFrame().assign(X=X_test, y=y_test)

model = BertForSequenceClassification.from_pretrained("wietsedv/bert-base-dutch-cased", num_labels=y.nunique()).to(device)
optimizer = torch.optim.Adam(model.parameters())
padding_idx = model.bert.embeddings.word_embeddings.padding_idx

for param in model.bert.parameters():
    param.requires_grad = False

metrics = defaultdict(list)
es_count = 0
for epoch in range(EPOCHS):
    batch_loss = 0
    model.train()
    for i, (Xb, yb, att_mask) in enumerate(iter_batches(train, bs=BATCH_SIZE, device=device)):
        model.zero_grad()
        out = model(Xb, attention_mask=att_mask, labels=yb)
        loss = out[0]
        loss.backward()
        optimizer.step()
        batch_loss += loss.data

    metrics["train_loss"].append(batch_loss / (i+1))

    batch_loss = 0
    n_corrects = 0
    predictions, targets = (np.empty_like(valid['y']) for _ in range(2))
    model.eval()
    for i, (Xb, yb, att_mask) in enumerate(iter_batches(valid, bs=64, device=device)):
        out = model(Xb, attention_mask=att_mask, labels=yb)
        loss = out[0].detach().cpu().numpy()
        batch_loss += loss.item()
        predictions[i*64:i*64+len(yb)] = out[1].argmax(1).detach().cpu().numpy()
        targets[i*64:i*64+len(yb)] = yb.cpu().numpy()

    metrics["valid_loss"].append(batch_loss / (i+1))
    metrics["accuracy"].append(accuracy_score(predictions, targets))

    print(f"Epoch {epoch}: train: {metrics['train_loss'][-1]:.4f} valid: {metrics['valid_loss'][-1]:.4f} {metrics['accuracy'][-1]:.2%}")

    if epoch > 0 and metrics['valid_loss'][-1] > min(metrics['valid_loss']):
        es_count += 1
    else:
        es_count = 0
    if epoch > ES_MIN_EPOCHS and es_count >= ES_PATIENCE:
        break

