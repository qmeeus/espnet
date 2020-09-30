#!/usr/bin/bash

[ $# -lt 1 ] && { echo "Usage: train_hybrid.sh outdir" && exit 1; }

outdir=$1

. setup.sh
. path.sh

asr_model=exp/train_unigram_5000_mono/transformer_12_6_2048_512_8_a.4_do.1/train/results/model.loss.best
hybrid_model=exp/train_hybrid_mono/transformer_12_6_2048_512_8_asched_do.1/dec-only/results/model.loss.best
teacher_model=distilbert-base-multilingual-cased

#asr_model=exp/train_unigram_5000_all/transformer_12_6_2048_512_8_a.4_do.1/train/results/model.loss.best
#teacher_model=examples/language-modeling/output/distilbert/checkpoint-134000

w2v_train.py \
  --v1 \
  --config conf/train/transformer_12_6_2048_512_8_asched_do.1.yaml \
  --ngpu 1 \
  --backend pytorch \
  --outdir $outdir/results \
  --debugmode 1 \
  --ctc_type builtin \
  --debugdir $outdir \
  --minibatches 0 \
  --verbose 1 \
  --resume \
  --model-class espnet.nets.pytorch_backend.e2e_xlmr_transformer:E2E \
  --dict data/lang_unigram/CGN_train_unigram_5000_units.txt \
  --train-json dump/CGN_train/deltafalse/data_unigram_5000.mono.json \
  --valid-json dump/CGN_valid/deltafalse/data_unigram_5000.mono.json \
  --teacher-model=$teacher_model \
  --enc-init $hybrid_model \
  --dec-init $hybrid_model
#  --enc-init $asr_model \
#  --freeze-encoder -1

