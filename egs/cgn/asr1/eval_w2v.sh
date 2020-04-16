#!/bin/bash -e

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

if [ -z "$output_dir" ]; then
  echo "output_dir not set. Exitting..." && exit 1
fi

set -u
set -o pipefail

mkdir -p $output_dir

dataset_tag="all"
dict=data/lang_word/CGN_train_word_units.txt
json_prefix="data_words"

if [ -z "$resume" ]; then
  echo "Not specified which model to evaluate"
  exit 1
fi

verbose=${verbose:-0}
test_features=dump/${test_set}/deltafalse

(

  w2v_recog.py \
    --v1 \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --backend pytorch \
    --outdir $output_dir/results \
    --tensorboard-dir $tensorboard_dir \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --ctc_type builtin \
    --debugdir $output_dir \
    --minibatches ${N} \
    --verbose $verbose \
    --enc-init ${resume} \
    --dec-init ${resume} \
    --test-json ${test_features}/${json_prefix}.${dataset_tag}.json \
  | tee $output_dir/test.log

) 3>&1 1>&2 2>&3 | tee $output_dir/test.err

