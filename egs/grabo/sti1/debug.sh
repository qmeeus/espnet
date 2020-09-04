#!/bin/bash -e

. setup.sh

jobdir=data/grabo_w2v/blocks/pp10/12blocks_exp0

python train_vectors.py \
    --data_dir $jobdir \
    --output_dir $jobdir \
    --feature_file data/grabo_w2v/features.h5 \
    --target_file data/grabo/target.csv \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --overwrite_cache \
    --logging_steps 100 \
    --eval_steps 100 \
    --overwrite_output_dir \
    --max_steps 1000 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 128 #\
#    > $jobdir/test.log
