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

dict=data/lang_word/CGN_train_word_units.txt
json_prefix="data_words"

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
      --config ${train_config} \
      --ngpu 0 \
      --outdir $outdir/results \
      --debugmode ${debugmode} \
      --dict ${dict} \
      --ctc_type builtin \
      --debugdir $outdir \
      --verbose $verbose \
      --enc-init $resume \
      --dec-init $resume \
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
