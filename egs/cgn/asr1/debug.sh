#!/bin/bash

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

# cd ../../..
# python espnet/data/dataset.py

asr_train.py \
 --config conf/train_mtlalpha1.0_sm.yaml \
 --ngpu 1 \
 --backend pytorch \
 --outdir exp/train_s__pytorch_vgglstmp/results \
 --tensorboard-dir tensorboard/train_s__pytorch_vgglstmp \
 --debugmode 1 \
 --dict data/lang_1char/train_s__units.txt \
 --ctc_type builtin \
 --debugdir exp/train_s__pytorch_vgglstmp \
 --minibatches 0 \
 --verbose 1 \
 --resume \
 --train-json dump/train_s_/deltafalse/data.json \
 --valid-json dump/dev_s/deltafalse/data.json
