#!/bin/bash -e

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

validset_tag=${validset_tag:-all}
dict=data/lang_word/CGN_train_word_units.txt
json_prefix="data_words"
# export TGT_WORDS=1

verbose=${verbose:-0}
# config_name=$(basename ${train_config%.*})
# target_name=$(echo $jsontrain | sed "s/data|\.json//g")
# expname="${train_set}_${config_name}${tag+_$tag}${target_name}"
train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse

i=0
PRETRAINED_MODEL=${resume:-exp/train_lstm_mtlalpha0.1_unigram_1000_data_all.2/results/model.loss.best}
# PRETRAINED_MODEL=exp/train_lstm_words_pretrained_curriculum/v1/train/mono/results/snapshot.ep.20
for dataset_tag in o ok mono all; do 

  outdir=$output_dir/$dataset_tag
  tb_dir=$tensorboard_dir/$dataset_tag
  mkdir -p $outdir

  OPTIONS="--enc-init $PRETRAINED_MODEL"

  if (( $i > 0 )); then
    OPTIONS="$OPTIONS --dec-init $PRETRAINED_MODEL"
  elif [ "$resume" ]; then
    OPTIONS="$OPTIONS --dec-init $PRETRAINED_MODEL"
  fi

  (
    w2v_train.py \
      --v1 \
      --config ${train_config} \
      --preprocess-conf ${preprocess_config} \
      --ngpu ${ngpu} \
      --backend pytorch \
      --outdir $outdir/results \
      --tensorboard-dir $tb_dir \
      --debugmode ${debugmode} \
      --dict ${dict} \
      --ctc_type builtin \
      --debugdir $outdir \
      --minibatches ${N} \
      --verbose $verbose \
      --train-json ${train_features}/${json_prefix}.${dataset_tag}.json \
      --valid-json ${dev_features}/${json_prefix}.${validset_tag}.json $OPTIONS \
    | tee $outdir/train.log 
  ) 3>&1 1>&2 2>&3 | tee $outdir/train.err

  PRETRAINED_MODEL=$outdir/results/model.loss.best
  i=$(( $i +  1 ))

done
