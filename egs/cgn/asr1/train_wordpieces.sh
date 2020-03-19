#!/bin/bash

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

output_dir=${output_dir:-exp/$exp}
train_config="conf/train_mtlalpha0.1.yaml"
tag="${tag:-unigram_5000}"
exp="${train_set}_$(basename ${train_config%.*})${tag+_$tag}"
dict="data/lang_char/train_s_unigram_5000_units.txt"
train_json="dump/train_s/deltafalse/data_unigram_5000_v2.json"
valid_json="dump/dev_s/deltafalse/data_unigram_5000.json"

mkdir -p exp/$exp

asr_train.py \
 --v1 \
 --config ${train_config} \
 --ngpu ${ngpu} \
 --backend pytorch \
 --outdir $output_dir/results \
 --tensorboard-dir tensorboard/$(basename $output_dir) \
 --debugmode 1 \
 --dict ${dict} \
 --ctc_type builtin \
 --debugdir $output_dir \
 --minibatches 0 \
 --verbose 0 \
 --train-json ${train_json} \
 --valid-json ${valid_json} \
 | tee $output_dir/train.log
