#!/bin/bash -xe

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -u
set -o pipefail

# =======================================================================================
#                                  PRELIMINARIES
# =======================================================================================
if [ $stage -le 0 ]; then
  echo "stage 0: prepare the data"
  # the script detects if a telephone comp is used and splits this into a separate set
  # later, studio and telephone speech can be combined for NNet training
  bash -xe local/cgn_data_prep.sh ${datadir} ${lang} ${comp} ${decodecomp} ${train_set} ${train_dev} ${tag}
fi

if [ ${stop_stage} -le 0 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                  PREPROCESSING
# =======================================================================================
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${train_dev}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 8 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    fulldata="train+dev_$tag"
    cp -r data/$train_set data/$fulldata

    # make a dev set
    utils/subset_data_dir.sh --first data/$fulldata 1000 data/${train_dev}
    n=$(($(wc -l < data/$fulldata/text) - 1000))
    utils/subset_data_dir.sh --last data/$fulldata ${n} data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

fi

if [ ${stop_stage} -le 1 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                       LETTERS
# =======================================================================================
dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then

    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh \
      --feat ${feat_tr_dir}/feats.scp \
      data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    
    data2json.sh \
      --feat ${feat_dt_dir}/feats.scp \
      data/${train_dev} ${dict} > ${feat_dt_dir}/data.json

fi

if [ ${stop_stage} -le 2 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                    WORDPIECES
# =======================================================================================

if [ $stage -le 3 ]; then

  echo "Train wordpiece model"
  model=unigram
  vocab_size=5000
  lang_dir=data/lang_char
  texts=$lang_dir/input.txt
  model_prefix=${lang_dir}/${train_set}_${model}_${vocab_size}
  vocab=${model_prefix}_units.txt

  mkdir -p $lang_dir
  cat data/${train_set}/text | cut -f 2- -d" " | iconv -f iso-8859-1 -t utf-8 -o data/lang_char/input.txt > $texts

  spm_train \
    --input=$texts \
    --vocab_size=$vocab_size \
    --model_type=${model} \
    --model_prefix=${model_prefix} \
    --input_sentence_size=100000000

  echo "<unk> 1" > $vocab

  spm_encode \
    --model=${model_prefix}.model \
    --output_format=piece < $texts \
    | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> $vocab

#   spm_encode \
#     --model=${model_prefix}.model \
#     --output_format=piece <(cat data/${train_dev}/text | cut -f2- -d" ") \
#     | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' \
#     >> $vocab

  data2json.sh \
      --feat ${feat_tr_dir}/feats.scp \
      --bpecode ${model_prefix}.model \
      data/${train_set} ${vocab} \
      > ${feat_tr_dir}/data_${model}_${vocab_size}.json

  data2json.sh \
    --feat ${feat_dt_dir}/feats.scp \
    --bpecode ${model_prefix}.model \
    data/${train_dev} ${vocab} \
    > ${feat_dt_dir}/data_${model}_${vocab_size}.json

fi

if [ ${stop_stage} -le 3 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                       TRAINING
# =======================================================================================
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then

    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --v1 \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --ctc_type builtin \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json

fi

if [ ${stop_stage} -le 4 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

echo "Done!"
