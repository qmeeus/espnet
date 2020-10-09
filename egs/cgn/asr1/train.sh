#!/bin/bash -e

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

OPTIONS=""

setup_target(){
  SCRIPT=asr_train.py
  model_class=espnet.nets.pytorch_backend.e2e_asr:E2E
  case $target in
    "char")
      dict=data/lang_1char/${train_set}_units.txt
      json_prefix="data"
      ;;
    "wordpiece")
      dict=data/lang_unigram/${train_set}_unigram_${vocab_size}_units.txt
      json_prefix="data_unigram_${vocab_size}"
      [[ "$train_config" = *"transformer"* ]] \
        && model_class=espnet.nets.pytorch_backend.e2e_asr_transformer:E2E \
        || model_class=espnet.nets.pytorch_backend.e2e_asr:E2E
      ;;
    "wordpiece_mlm")
      dict=data/lang_unigram/${train_set}_unigram_${vocab_size}_units.txt
      json_prefix="data_unigram_${vocab_size}"
      model_class=espnet.nets.pytorch_backend.e2e_student:E2E
      ;;
    "word")
      dict=${dict:-data/lang_word/${train_set}_word_units.txt}
      json_prefix="${json_prefix:-data_words}"
      SCRIPT=w2v_train.py
      [[ "$train_config" = *"transformer"* ]] \
        && model_class=espnet.nets.pytorch_backend.e2e_w2v_transformer:E2E \
        || model_class=espnet.nets.pytorch_backend.e2e_w2v:E2E
      ;;
    "pos")
      dict=data/lang_word/${train_set}_pos_units.txt
      json_prefix="data_pos_300"
      ;;
    "vector")
      dict=data/lang_unigram/${train_set}_unigram_${vocab_size}_units.txt
      # json_prefix="data_vectors_bert_dutch"
      json_prefix="data_unigram_${vocab_size}"
      SCRIPT=w2v_train.py
      model_class=espnet.nets.pytorch_backend.e2e_xlmr_transformer:E2E
      ;;
    *)
      echo "Invalid target: $target" && exit 1
      ;;
  esac

}

setup_target
verbose=${verbose:-20}
train_features=dump/${train_set}/deltafalse
dev_features=dump/${dev_set}/deltafalse
validset_tag=${validset_tag:-$dataset_tag}
train_json=${train_features}/${json_prefix}.${dataset_tag}.json
valid_json=${dev_features}/${json_prefix}.${validset_tag}.json

OPTIONS="--dict $dict --train-json $train_json --valid-json $valid_json"
[ -z "${enc_init}" ] || OPTIONS="$OPTIONS --enc-init $enc_init"
[ -z "$dec_init" ] || OPTIONS="$OPTIONS --dec-init $dec_init"
[ -z "$tensorboard_dir" ] || OPTIONS="$OPTIONS --tensorboard-dir $tensorboard_dir"
[ -z "$freeze_encoder" ] || OPTIONS="$OPTIONS --freeze-encoder $freeze_encoder"
[ -z "$model_class" ] || OPTIONS="$OPTIONS --model-class $model_class"
[ -z "$emb_dim" ] || OPTIONS="$OPTIONS --emb-dim $emb_dim"
[ -z "$emb_path" ] || OPTIONS="$OPTIONS --emb-path $emb_path"

mkdir -p $output_dir

(

  $SCRIPT \
    --v1 \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --backend pytorch \
    --outdir $output_dir/results \
    --debugmode ${debugmode} \
    --ctc_type builtin \
    --debugdir $output_dir \
    --minibatches ${N} \
    --verbose $verbose \
    --resume ${resume} \
    $OPTIONS #\
    # | tee $output_dir/train.log

)  #3>&1 1>&2 2>&3 | tee $output_dir/train.err

