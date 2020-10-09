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
verbose=20       # verbose option
resume=         # Resume the training from snapshot
enc_init=
dec_init=

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train_mtlalpha0.1.yaml
lm_config=
decode_config=conf/decode_ctcweight0.1.yaml
exp_config=
json_prefix=
dict=
emb_dim=
emb_path=

# # rnnlm related
# use_wordlm=true     # false means to train/use a character LM
# lm_vocabsize=100    # effective only for word LMs
# lmtag=              # tag for managing LMs
# lm_resume=          # specify a snapshot file to resume LM training
freeze_encoder=

# decoding parameter
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
recog_set="a b f g h i j k l m n o"

# exp tag
tag=                  # tag for managing experiments.
output_dir=
tensorboard_dir=
target="wordpiece"
dataset_tag=
validset_tag=
curriculum="ok mono all"

# data
cgn_root=/users/spraak/spchdata/cgn
# cgn_root=/esat/spchdisk/scratch/qmeeus/data/cgn/preprocessed
lang="vl"
vocab_size=

# tag="xs"
# comp="o;k"
# decodecomp="o;k"
# train_set="train_xs"
# dev_set="dev_xs"
# test_set="test_xs"

# tag="s"
# comp="i;j;k;l;m;n;o"
# decodecomp="i;j;k;l;m;n;o"
# train_set="train_s"
# dev_set="dev_s"
# test_set="test_s"

# tag="m"
# comp="o;k;l;j;m;n;g;f;b;h;a;i"
# decodecomp="o;k;l;j;m;n;g;f;b;h;a;i"
train_set="CGN_train"
dev_set="CGN_valid"
test_set="CGN_test"
