#!/bin/bash


# general configuration
backend=pytorch
stage=0         # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
gpus=1
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1       # verbose option
resume=         # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train_mtlalpha0.1.yaml
lm_config=
decode_config=conf/decode.yaml

# # rnnlm related
# use_wordlm=true     # false means to train/use a character LM
# lm_vocabsize=100    # effective only for word LMs
# lmtag=              # tag for managing LMs
# lm_resume=          # specify a snapshot file to resume LM training

# decoding parameter
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
datadir=/users/spraak/spchdata/cgn
cgn_root=/esat/spchdisk/scratch/qmeeus/data/cgn/preprocessed
lang="vl"

# comp="o;k"
# decodecomp="o;k"
# dict=data/lang_1char/train_s_units.txt
# train_set="train_s"
# dev_set="dev_s"

comp="o;k;l;j;m;n;g;f;b;h;a;i"
decodecomp="o;k;l;j;m;n;g;f;b;h;a;i"
train_set="train_m"
dev_set="dev_m"
dict=data/lang_1char/${train_set}_units.txt
wpdict=data/lang_char/${train_set}_unigram_5000_units.txt

# exp tag
tag="s"             # tag for managing experiments.
output_dir=""

