#!/bin/bash

#TAG=transformer_bertje_decoder
TAG=debug_transformer
#PRETRAINED_ASR_MODEL=exp/train_unigram_5000_mono/transformer_12_6_2048_512_8_a.4_do.1/train/results/model.loss.best
PRETRAINED_ASR_MODEL=exp/train_unigram_5000_mono/transformer_12_6_2048_512_8_a.4_do.1/train/results/model.acc.best

. path.sh
#./train.sh --target vector --ngpu 1 --output_dir exp/debug/xmlr/train --train_config conf/train/transformer_6_3_512_128_4_do.1.yaml --dataset_tag o --tag debug

#./eval_w2v.sh --target vector --ngpu 0 --train_config conf/train/transformer_12_6_2048_512_8_do.1.yaml --resume exp/train_hybrid_mono/transformer_12_6_2048_512_8_do.1/dec-only/results/model.loss.best --tag debug --output_dir exp/debug/w2v/evaluate

#./train.sh --target word --ngpu 1 --output_dir exp/debug/w2v/train --train_config conf/train/transformer_6_3_512_128_4_do.1.yaml --dataset_tag o --tag debug

#./train.sh --target word --ngpu 0 --output_dir exp/debug/w2v/train --train_config conf/train/lstmp_6_3_800_a.4_do.1.1_wd5.yaml --dataset_tag mono --tag debug

TAG=transformer_bertje_decoder
./train.sh \
  --target wordpiece_mlm \
  --ngpu 1 \
  --verbose 10 \
  --output_dir exp/debug/asr/train/$TAG \
  --train_config conf/train/$TAG.retrain.yaml \
  --dataset_tag o \
  --vocab-size 5000 \
  --tag $TAG \
  --freeze-encoder -1 \
  --enc-init exp/train_unigram_5000_mono/transformer_bertje_decoder/train/results/model.loss.best

#./train.sh \
#  --target wordpiece \
#  --ngpu 1 \
#  --output_dir exp/debug/asr/train \
#  --train_config conf/train/transformer_12_6_2048_512_8_asched_do.1.yaml \
#  --dataset_tag o \
#  --vocab-size 5000 \
#  --tag transformer_12_6_2048_512_8_asched_do.1 \
#  --enc-init $PRETRAINED_ASR_MODEL --dec-init $PRETRAINED_ASR_MODEL

#w2v_train.py \
#  --v1 \
#  --config conf/train/transformer_12_6_2048_512_8_do.1.yaml \
#  --ngpu 1 \
#  --backend pytorch \
#  --outdir exp/train_bert_dutch_mono/transformer_12_6_2048_512_8_do.1/dec-only/results \
#  --debugmode 1 \
#  --ctc_type builtin \
#  --debugdir exp/train_bert_dutch_mono/transformer_12_6_2048_512_8_do.1/dec-only \
#  --minibatches 0 \
#  --verbose 1 \
#  --resume \
#  --dict none \
#  --train-json dump/CGN_train/deltafalse/data_vectors_bert_dutch.mono.json \
#  --valid-json dump/CGN_valid/deltafalse/data_vectors_bert_dutch.mono.json \
#  --enc-init exp/train_unigram_5000_mono/transformer_12_6_2048_512_8_a.4_do.1/train/results/model.loss.best \
#  --freeze-encoder -1 \
#  --model-class espnet.nets.pytorch_backend.e2e_xlmr_transformer:E2E

#w2v_train.py \
#  --v1 \
#  --config conf/train/transformer_12_6_2048_512_8_a.4_do.1.yaml \
#  --ngpu 1 \
#  --backend pytorch \
#  --outdir exp/debug/train_hybrid/results \
#  --debugmode 1 \
#  --ctc_type builtin \
#  --debugdir exp/debug/train_hybrid \
#  --minibatches 0 \
#  --verbose 1 \
#  --resume \
#  --dict data/lang_unigram/CGN_train_unigram_5000_units.txt\
#  --train-json dump/CGN_train/deltafalse/data_unigram_5000.mono.json \
#  --valid-json dump/CGN_valid/deltafalse/data_unigram_5000.mono.json \
#  --enc-init exp/train_unigram_5000_mono/transformer_12_6_2048_512_8_a.4_do.1/train/results/model.loss.best \
#  --freeze-encoder -1 \
#  --model-class espnet.nets.pytorch_backend.e2e_xlmr_transformer:E2E

