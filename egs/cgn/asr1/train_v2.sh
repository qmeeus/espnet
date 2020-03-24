#!/bin/bash

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

case $target in
  "char")
    dict=data/lang_1char/${train_set}_units.txt
    jsonfile="data${dataset_tag}.json"
    ;;
  "wordpiece")
    dict=data/lang_char/${train_set}_unigram_${vocab_size}_units.txt
    jsonfile="data_unigram_${vocab_size}${dataset_tag}.json"
    ;;
  "word")
    dict=data/lang_word/${train_set}_word_units.txt
    jsonfile="data_word${dataset_tag}.json"
    ;;
  "pos")
    dict=data/lang_word/${train_set}_pos_units.txt
    jsonfile="data_pos_300${dataset_tag}.json"
    ;;
  *)
    echo "Invalid target: $target" && exit 1
    ;;
esac

# config_name=$(basename ${train_config%.*})
# target_name=$(echo $jsonfile | sed "s/data|\.json//g")
# expname="${train_set}_${config_name}${tag+_$tag}${target_name}"
train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse

# output_dir=${output_dir:-exp/$expname}
mkdir -p $output_dir

asr_train.py \
  --config ${train_config} \
  --ngpu ${ngpu} \
  --backend pytorch \
  --outdir $output_dir/results \
  --tensorboard-dir tensorboard/$(basename $output_dir) \
  --debugmode ${debugmode} \
  --dict ${dict} \
  --ctc_type builtin \
  --debugdir $output_dir \
  --minibatches ${N} \
  --verbose 0 \
  --resume ${resume} \
  --train-json ${train_features}/$jsonfile \
  --valid-json ${dev_features}/$jsonfile \
  | tee $output_dir/train.log
