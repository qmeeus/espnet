#!/bin/bash -e

[ $# -lt 1 ] && { echo Missing args! && exit 1 ;}

. setup.sh

pids=()
for jobdir in $(cat $1); do
    (
        python train_vectors.py \
            --data_dir $jobdir \
            --output_dir $jobdir \
            --feature_file data/features.h5 \
            --target_file data/encoded_target.csv \
            --do_train \
            --do_eval \
            --do_predict \
            --evaluate_during_training \
            --overwrite_cache \
            --logging_steps 2000 \
            --eval_steps 2000 \
            --overwrite_output_dir \
            --max_steps 20000 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 128 \
            > $jobdir/test.log
    ) &
    pids+=($!)
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
if [ ${i} -gt 0  ]; then
    echo "$0: ${i} background jobs are failed."
    exit 1
fi

