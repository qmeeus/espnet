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

dict=${dict:-data/lang_word/CGN_train_word_units.txt}
json_prefix=${json_prefix:-data_words}
model_class=espnet.nets.pytorch_backend.e2e_w2v_transformer:E2E
[ "$target" == vector ] && model_class=espnet.nets.pytorch_backend.e2e_xlmr_transformer:E2E
model_class=espnet.nets.pytorch_backend.e2e_mlm_transformer:E2E

if [ -z "$resume" ]; then
  echo "Not specified which model to evaluate"
  exit 1
fi

verbose=${verbose:-0}
test_features=dump/${test_set}/deltafalse

pids=() # initialize pids
for dataset in $(ls $test_features/${json_prefix}.?.json); do
  (
    tag=$(basename $dataset | awk -F "." '{ print $2 }')
    outdir=$output_dir/$tag
    mkdir -p $outdir

    w2v_recog.py \
      --v1 \
      --task ${target:-word} \
      --model-class ${model_class} \
      --config ${train_config} \
      --ngpu 1 \
      --outdir $outdir/results \
      --debugmode ${debugmode} \
      --dict ${dict} \
      --ctc_type builtin \
      --debugdir $outdir \
      --verbose $verbose \
      --resume $resume \
      --test-json $dataset \
    > $outdir/test.log 2> $outdir/test.err

  ) &

  pids+=($!) # store background pids
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
if [ ${i} -gt 0 ]; then
    echo "$0: ${i} background jobs are failed."
    exit 1
fi
