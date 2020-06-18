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
      dict=data/lang_unigram/${train_set}_unigram_${vocab_size}_units.txt
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

curriculum=${curriculum:-"o ok mono all"}
verbose=${verbose:-0}
train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse

i=0
for dataset_tag in $curriculum; do

  outdir=$output_dir/$dataset_tag
  mkdir -p $outdir

  OPTIONS=""
  if (( $i > 0 )); then
    OPTIONS="--enc-init $PREV_MODEL --dec-init $PREV_MODEL"
  else
    if ! [ -z $resume ]; then
      OPTIONS="--resume $resume"
    else
      [ -z $enc_init ] || OPTIONS="--enc-init $enc_init"
      [ -z $dec_init ] || OPTIONS="$OPTIONS --dec-init $dec_init"
    fi
  fi

  if ! [ -z "$tensorboard_dir" ]; then
    OPTIONS="$OPTIONS --tensorboard-dir $tensorboard_dir/$dataset_tag"
  fi

  valid_set=${validset_tag:-$dataset_tag}

  (
    $SCRIPT \
      --v1 \
      --config ${train_config} \
      --ngpu ${ngpu} \
      --backend pytorch \
      --outdir $outdir/results \
      --debugmode ${debugmode} \
      --dict ${dict} \
      --ctc_type builtin \
      --debugdir $outdir \
      --minibatches ${N} \
      --verbose $verbose \
      --train-json ${train_features}/${json_prefix}.${dataset_tag}.json \
      --valid-json ${dev_features}/${json_prefix}.${valid_set}.json $OPTIONS \
    | tee $outdir/train.log
  ) 3>&1 1>&2 2>&3 | tee $outdir/train.err

  PREV_MODEL=$outdir/results/model.loss.best
  i=$(( $i +  1 ))

done
