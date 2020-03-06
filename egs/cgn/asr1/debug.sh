#!/bin/bash

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

asr_train.py \
  --config conf/train_mtlalpha1.0.yaml \
  --ngpu 1 \
  --backend pytorch \
  --outdir exp/train_nodev_pytorch_train_mtlalpha1.0/results \
  --tensorboard-dir tensorboard/train_nodev_pytorch_train_mtlalpha1.0 \
  --debugmode 1 \
  --dict data/lang_1char/train_nodev_units.txt \
  --ctc_type builtin \
  --debugdir exp/train_nodev_pytorch_train_mtlalpha1.0 \
  --minibatches 0 \
  --verbose 1 \
  --resume \
  --train-json dump/train_nodev/deltafalse/data.json \
  --valid-json dump/train_dev/deltafalse/data.json
