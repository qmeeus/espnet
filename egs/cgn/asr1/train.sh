#!/bin/bash -e

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

validset_tag=${validset_tag:-$dataset_tag}

case $target in
  "char")
    dict=data/lang_1char/${train_set}_units.txt
    json_prefix="data"
    ;;
  "wordpiece")
    dict=data/lang_char/${train_set}_unigram_${vocab_size}_units.txt
    json_prefix="data_unigram_${vocab_size}"
    ;;
  "word")
    dict=data/lang_word/${train_set}_word_units.txt
    json_prefix="data_word"
    ;;
  "pos")
    dict=data/lang_word/${train_set}_pos_units.txt
    json_prefix="data_pos_300"
    ;;
  *)
    echo "Invalid target: $target" && exit 1
    ;;
esac


# config_name=$(basename ${train_config%.*})
# target_name=$(echo $jsontrain | sed "s/data|\.json//g")
# expname="${train_set}_${config_name}${tag+_$tag}${target_name}"
train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse

# output_dir=${output_dir:-exp/$expname}
mkdir -p $output_dir

asr_train.py \
  --v1 \
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
  --train-json ${train_features}/${json_prefix}${dataset_tag}.json \
  --valid-json ${dev_features}/${json_prefix}${validset_tag}.json \
  | tee $output_dir/train.log
