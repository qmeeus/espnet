#!/bin/bash

. cmd.sh
. path.sh

nj=1
model_dir=${1:-exp/CGN_train_pytorch_train_pytorch_transformer_lr5.0_ag8.v2_specaug}
featdir=dump/grabo_patience/nopitch
# featdir=dump/CGN_test/nopitch
# subsets="grabo"
subsets="grabo patience"
# subsets="a b e f g h i j k l m n o"
# subsets="a o"

pids=()
for subset in $subsets; do

  # (
    decode_dir=$model_dir/decode_${subset}_model.val5.avg.best_decode_lm
    ${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
      asr_recog.py \
        --config conf/decode.yaml \
        --ngpu 1 \
        --backend pytorch \
        --batchsize 0 \
        --recog-json $featdir/data.${subset}.json \
        --result-label $decode_dir/data.json \
        --model $model_dir/results/model.val5.avg.best \
        --api v2
  # ) &
  # pids+=($!)
done

# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0  ] && echo "$0: ${i} background jobs are failed." && false
