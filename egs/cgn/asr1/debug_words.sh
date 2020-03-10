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

exp="${2:-train_s__pytorch_vggblstmp_words_mtalpha0.1}"

asr_train.py \
 --config conf/train_mtlalpha0.5.yaml \
 --ngpu 1 \
 --backend pytorch \
 --outdir exp/$exp/results \
 --tensorboard-dir tensorboard/$exp \
 --debugmode 1 \
 --dict data/lang_1word/train_s__units.txt \
 --ctc_type builtin \
 --debugdir exp/train_s__pytorch_vgglstmp \
 --minibatches 0 \
 --verbose 1 \
 --train-json dump/train_s_words/data.json \
 --valid-json dump/dev_s_words/data.json
