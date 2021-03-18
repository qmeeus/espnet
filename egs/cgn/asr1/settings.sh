#!/bin/bash


# general configuration
backend=pytorch
stage=4       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train_mtlalpha0.5.yaml
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
comp="o;k"
decodecomp="o;k"  # ;l;j;m;n;g;f;b;h;a;i

# exp tag
tag="" # tag for managing experiments.

# datasets
# train_set="train_s_"
# train_dev="dev_s"
train_set=CGN_train
train_dev=CGN_valid

