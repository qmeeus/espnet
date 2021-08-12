#!/bin/bash

. path.sh

train_config=$1
preprocess_config=conf/specaug.yaml
train_set=train_960
backend=pytorch
expdir=exp/${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
use_valbest_average=true
n_average=5


if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
  [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
  [[ $(get_yaml.py ${train_config} etype) = transformer ]] || \
  [[ $(get_yaml.py ${train_config} dtype) = transformer ]]; then

  # Average ASR models
  if ${use_valbest_average}; then
    recog_model=model.val${n_average}.avg.best
    opt="--log ${expdir}/results/log"
  else
    recog_model=model.last${n_average}.avg.best
    opt="--log"
  fi
  average_checkpoints.py \
    ${opt} \
    --backend ${backend} \
    --snapshots ${expdir}/results/snapshot.ep.* \
    --out ${expdir}/results/${recog_model} \
    --num ${n_average}

  if [ $? -eq 0 ]; then
    echo "Remove intermediary snapshots"
    ls -halt ${expdir}/results/snapshot.ep.* | tail -n+2 | rev | cut -d" " -f1 | rev | xargs rm
  fi

fi

# vim:ts=2:sw=2:et
