#!/bin/bash

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

feat_tr_dir=dump/${train_set}/deltafalse
feat_dt_dir=dump/${train_dev}/deltafalse

asr_train.py \
  --config ${train_config} \
  --ngpu ${ngpu} \
  --gpus ${gpus} \
  --backend ${backend} \
  --outdir ${expdir}/results \
  --tensorboard-dir tensorboard/${expname} \
  --debugmode ${debugmode} \
  --dict ${dict} \
  --ctc_type builtin \
  --debugdir ${expdir} \
  --minibatches ${N} \
  --verbose ${verbose} \
  --resume ${resume} \
  --train-json ${feat_tr_dir}/data.json \
  --valid-json ${feat_dt_dir}/data.json
> ${expdir}/train.log 2>&1
