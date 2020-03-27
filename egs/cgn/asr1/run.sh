#!/bin/bash -ex

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
  local/cgn_data_prep.sh ${datadir} ${lang} ${comp} ${decodecomp} ${train_set} ${dev_set} ${tag}

fi

if [ ${stop_stage} -le 0 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                  PREPROCESSING
# =======================================================================================
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${dev_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 8 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

fi

if [ ${stop_stage} -le 1 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                               TRAIN / DEV / TEST
# =======================================================================================
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}/delta${do_delta}; mkdir -p ${feat_dt_dir}
feat_te_dir=${dumpdir}/${test_set}/delta${do_delta}; mkdir -p ${feat_te_dir}

if [ ${stage} -le 2 ]; then
    echo "stage 2: Dataset splits"

    fulldata="data_${tag}_all"
    train_dev="train+dev_${tag}"
    cp -r data/$train_set data/$fulldata
    # n=$(wc -l < data/$fulldata/text)
    # ntest=$(python -c "print(int($n * .3))")
    # ndev=$(python -c "print(int($n * .1))")
    # ntrain=$(( $n - $ntest - $ndev ))

    # echo "Subset sizes: TRAIN: $ntrain    DEV: $ndev    TEST: $ntest    TOTAL: $n"

    # train / test split
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 30 data/$fulldata data/$train_dev data/$test_set
    # rm -r data/$fulldata

    # train / dev split
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 30 data/$train_dev data/$train_set data/$test_set
    # rm -r data/train_dev

    # # train / test split
    # utils/subset_data_dir.sh --first data/$fulldata $ntest data/${test_set}
    # utils/subset_data_dir.sh --last data/$fulldata $(( $n - $ntest )) data/${train_dev}
    # # rm -r data/$fulldata

    # # train / dev split
    # utils/subset_data_dir.sh --first data/${train_dev} $ndev data/${dev_set}
    # utils/subset_data_dir.sh --last data/${train_dev} $ntrain data/${train_set}
    # # rm -r data/train_dev

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
      data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
      data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
      data/${test_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/test ${feat_te_dir}

fi

if [ ${stop_stage} -le 2 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                       LETTERS
# =======================================================================================
dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 3 ]; then

    echo "stage 3: Dictionary and Json Data Preparation"
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
      data/${dev_set} ${dict} > ${feat_dt_dir}/data.json

    data2json.sh \
      --feat ${feat_te_dir}/feats.scp \
      data/${test_set} ${dict} > ${feat_te_dir}/data.json

fi

if [ ${stop_stage} -le 3 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                    WORDPIECES
# =======================================================================================

if [ $stage -le 4 ]; then

  echo "Stage 4: Train wordpiece model"
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
#     --output_format=piece <(cat data/${dev_set}/text | cut -f2- -d" ") \
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
    data/${dev_set} ${vocab} \
    > ${feat_dt_dir}/data_${model}_${vocab_size}.json

  data2json.sh \
    --feat ${feat_te_dir}/feats.scp \
    --bpecode ${model_prefix}.model \
    data/${test_set} ${vocab} \
    > ${feat_te_dir}/data_${model}_${vocab_size}.json

fi

if [ ${stop_stage} -le 4 ]; then
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

if [ ${stage} -le 5 ]; then

    echo "stage 5: Network Training"
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

if [ ${stop_stage} -le 5 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                       DECODING
# =======================================================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    n_jobs=8

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${n_jobs} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${n_jobs} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${n_jobs}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            ${recog_opts}

        score_sclite.sh ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    if [ ${i} -gt 0 ]; then 
        echo "$0: ${i} background jobs are failed."
        exit 1
    fi
fi



echo "Done!"
