#!/bin/bash

./train.sh \
  --ngpu 1 \
  --target xlmr \
  --output_dir exp/debug/xmlr/train \
  --train_config conf/train/transformer_6_3_512_128_4_do.1.yaml \
  --dataset_tag o \
  --tag debug

#./eval_w2v.sh \
#  --ngpu 0 \
#  --train_config conf/train/transformer_12_6_2048_512_8_do.1.yaml \
#  --target word \
#  --resume exp/train_word_vectors_mono/transformer_12_6_2048_512_8_do.1/dec-only/results/model.loss.best \
#  --tag debug \
#  --output_dir exp/debug/w2v/evaluate

#./train.sh \
#  --ngpu 1 \
#  --target word \
#  --output_dir exp/debug/w2v/train \
#  --train_config conf/train/transformer_6_3_512_128_4_do.1.yaml \
#  --dataset_tag o \
#  --tag debug

#./train.sh \
#  --ngpu 0 \
#  --target word \
#  --output_dir exp/debug/w2v/train \
#  --train_config conf/train/lstmp_6_3_800_a.4_do.1.1_wd5.yaml \
#  --dataset_tag mono \
#  --tag debug

