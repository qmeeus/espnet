#!/bin/bash

[ $# -lt 2 ] && { echo Missing args! && exit 1 ;}

data_dir=$1
output_dir=$2

. setup.sh

python train_vectors.py \
    --data_dir $data_dir \
    --output_dir $output_dir \
    --feature_file data/grabo_w2v/features.h5 \
    --target_file data/grabo/target.csv \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --overwrite_cache \
    --logging_steps 2000 \
    --eval_steps 2000 \
    --overwrite_output_dir \
    --max_steps 10000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128

