#!/bin/bash -e

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

setup_target(){
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
      json_prefix="data_words"
      ;;
    "pos")
      dict=data/lang_word/${train_set}_pos_units.txt
      json_prefix="data_pos_300"
      ;;
    *)
      echo "Invalid target: $target" && exit 1
      ;;
  esac

  if [ "$target" == "word" ]; then
    SCRIPT=w2v_train.py
  else
    SCRIPT=asr_train.py
  fi
}

setup_target
verbose=${verbose:-0}
train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse
validset_tag=${validset_tag:-$dataset_tag}

OPTIONS=""
if ! [ -z "${enc_init}" ]; then
  OPTIONS="--enc-init $enc_init"
fi

if ! [ -z "$dec_init" ]; then
  OPTIONS="$OPTIONS --dec-init $dec_init"
fi

if ! [ -z "$tensorboard_dir" ]; then
  OPTIONS="$OPTIONS --tensorboard-dir $tensorboard_dir"
fi

mkdir -p $output_dir

(

  $SCRIPT \
    --v1 \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --backend pytorch \
    --outdir $output_dir/results \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --ctc_type builtin \
    --debugdir $output_dir \
    --minibatches ${N} \
    --verbose $verbose \
    --resume ${resume} \
    --train-json ${train_features}/${json_prefix}.${dataset_tag}.json \
    --valid-json ${dev_features}/${json_prefix}.${validset_tag}.json $OPTIONS \
    | tee $output_dir/train.log

)  3>&1 1>&2 2>&3 | tee $output_dir/train.err

