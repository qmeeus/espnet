#!/bin/bash -e

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

#set -u
set -o pipefail

setup_target(){
  case $target in
    "char")
      dict=data/lang_1char/${train_set}_units.txt
      json_prefix="data"
      ;;
    "wordpiece")
      dict=data/lang_char/${train_set}_unigram_${vocab_size}_units.txt
      json_prefix="data_unigram_${vocab_size}"
      ;;
    "word")
      dict=data/lang_word/${train_set}_word_units.txt
      json_prefix="data_words"
      ;;
    "pos")
      dict=data/lang_word/${train_set}_pos_units.txt
      json_prefix="data_pos_300"
      ;;
    *)
      echo "Invalid target: $target" && exit 1
      ;;
  esac

}

setup_target
train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse
test_features=dump/${test_set}/deltafalse
#mkdir -p $output_dir

n_jobs=1
pids=() # initialize pids
for rtask in ${recog_set}; do
  (
    decode_dir=${output_dir}/${rtask}
    mkdir -p $decode_dir

    asr_recog.py \
      --ngpu ${ngpu:-0} \
      --api v2 \
      --config ${decode_config} \
      --backend ${backend} \
      --debugmode ${debugmode} \
      --verbose ${verbose} \
      --recog-json ${test_features}/${json_prefix}.${rtask}.json \
      --result-label ${output_dir}/results.${rtask}.json \
      --model ${recog_model} \
      2>&1 | tee ${decode_dir}/decode.log

    #score_sclite.sh ${decode_dir} ${dict}

  ) &
  pids+=($!) # store background pids
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
if [ ${i} -gt 0 ]; then
    echo "$0: ${i} background jobs are failed."
    exit 1
fi
