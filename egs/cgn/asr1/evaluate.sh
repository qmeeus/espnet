#!/bin/bash -xe

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

case $target in
  "char")
    dict=data/lang_1char/${train_set}_units.txt
    jsonfile="data${dataset_tag}.json"
    ;;
  "wordpiece")
    dict=data/lang_char/${train_set}_unigram_${vocab_size}_units.txt
    jsonfile="data_unigram_${vocab_size}${dataset_tag}.json"
    ;;
  "word")
    dict=data/lang_word/${train_set}_word_units.txt
    jsonfile="data_word${dataset_tag}.json"
    ;;
  "pos")
    dict=data/lang_word/${train_set}_pos_units.txt
    jsonfile="data_pos_300${dataset_tag}.json"
    ;;
  *)
    echo "Invalid target: $target" && exit 1
    ;;
esac

train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse
test_features=dump/${test_set}/deltafalse
mkdir -p $output_dir

n_jobs=1
pids=() # initialize pids
for rtask in ${recog_set}; do
(
    decode_dir=decode_$(basename ${decode_config%.*})_$(basename ${rtask%.*})

    # split data
    # splitjson.py --parts ${n_jobs} ${rtask}

    #### use CPU for decoding
    asr_recog.py \
      --config ${decode_config} \
      --ngpu 1 \
      --backend ${backend} \
      --debugmode ${debugmode} \
      --verbose ${verbose} \
      --recog-json ${test_features}/$jsonfile \
      --result-label ${output_dir}/$(basename ${rtask%.*}).json \
      --model ${recog_model} \
      2>&1 | tee $output_dir/decode.log

    score_sclite.sh ${expdir}/${decode_dir} ${dict}

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
if [ ${i} -gt 0 ]; then 
    echo "$0: ${i} background jobs are failed."
    exit 1
fi