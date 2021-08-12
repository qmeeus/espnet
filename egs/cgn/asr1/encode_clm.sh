#!/bin/bash -e

. path.sh
. cmd.sh

# dataset=grabo
dataset=patience
expdir=exp/CGN_train_pytorch_train_pytorch_transformer_lr5.0_ag8.v2_specaug
decode_dir=decode_${dataset}_model.val5.avg.best_decode_lm
feat_recog_dir=dump/grabo_patience/nopitch

splitjson.py --parts 8 ${feat_recog_dir}/data.${dataset}.json
${decode_cmd} JOB=1:8 ${expdir}/${decode_dir}/log/decode.JOB.log \
  asr_encode.py \
    --config conf/decode.yaml \
    --ngpu 0 \
    --backend pytorch \
    --batchsize 0 \
    --recog-json ${feat_recog_dir}/split8utt/data.${dataset}.JOB.json \
    --result-label ${expdir}/${decode_dir}/results.JOB.json \
    --model ${expdir}/results/model.val5.avg.best \
    --api v1
