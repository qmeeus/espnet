#!/bin/bash -e

. setup.sh
. path.sh
. cmd.sh
. settings.sh
. utils/parse_options.sh

set -x
set -u
set -o pipefail

data_dir=data/CGN_ALL

# =======================================================================================
#                                  PRELIMINARIES
# =======================================================================================
if [ $stage -le 0  ]; then
  echo "stage 0: prepare the data"
  python local/prep_utterances.py
  python local/RAW_prepare_cgn_annot.py \
    --annot-file ${data_dir}/annotations.csv \
    --file-list data/datafiles.csv \
    --use-existing-annot \
    --use-existing-file-registry
  utils/validate_data_dir.sh --no-feats ${data_dir}
fi

if [ ${stop_stage} -le 0  ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                  PREPROCESSING
# =======================================================================================
if [ ${stage} -le 1  ]; then
  echo "stage 1: Feature Generation"
  fbankdir=fbank
  # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
  #steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 8 --write_utt2num_frames true \
  #  ${data_dir} exp/make_fbank/CGN_ALL ${fbankdir}
  steps/make_fbank.sh --cmd "$train_cmd" --nj 8 --write_utt2num_frames true \
    ${data_dir} exp/make_fbank/CGN_ALL ${fbankdir}
fi

if [ ${stop_stage} -le 1  ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                               TRAIN / DEV / TEST
# =======================================================================================
if [ ${stage} -le 2  ]; then
  echo "stage 2: Dataset splits"
  # The following line has to be done in a normal env bc path.sh messes up locales
  # python local/RAW_split_subsets.py ${data_dir} --annot-file annotations.csv --prefix CGN
  compute-cmvn-stats scp:data/CGN_train/feats.scp data/CGN_train/cmvn.ark

  for subset in train valid test; do
    #feature_dir=${dumpdir}/CGN_${subset}/delta${do_delta}
    feature_dir=${dumpdir}/CGN_${subset}/nopitch
    mkdir -p $feature_dir
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta ${do_delta} \
        data/CGN_${subset}/feats.scp data/CGN_train/cmvn.ark exp/dump_feats/${subset} ${feature_dir}
  done
fi

if [ ${stop_stage} -le 2  ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi

# =======================================================================================
#                                    WORDPIECES
# =======================================================================================
if [ $stage -le 3 ]; then

  echo "Stage 3: Train wordpiece model"
  model=unigram
  vocab_size=${vocab_size:-5000}
  lang_dir=data/lang_${model}
  texts=$lang_dir/input.txt
  model_prefix=${lang_dir}/${train_set}_${model}_${vocab_size}
  vocab=${model_prefix}_units.txt

  mkdir -p $lang_dir
  cat data/${train_set}/text | cut -f 2- -d" " | iconv -f iso-8859-1 -t utf-8 -o ${lang_dir}/input.txt > $texts

  spm_train \
    --input=$texts \
    --user_defined_symbols=0,1,2,3,4,5,6,7,8,9 \
    --vocab_size=$vocab_size \
    --model_type=${model} \
    --model_prefix=${model_prefix} \
    --input_sentence_size=100000000

  echo "<unk> 1" > $vocab
  spm_encode \
    --model=${model_prefix}.model \
    --output_format=piece < $texts \
    | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> $vocab

  for subset in train valid test; do
    # feature_dir=dump/CGN_${subset}/delta${do_delta}
    feature_dir=dump/CGN_${subset}/nopitch
    data2json.sh \
      --feat ${feature_dir}/feats.scp \
      --bpecode ${model_prefix}.model \
      data/CGN_${subset} ${vocab} \
      > ${feature_dir}/data_${model}_${vocab_size}.json
  done

  python local/RAW_split_datasets_components.py data_${model}_${vocab_size}
  python local/RAW_split_datasets_components.py data_${model}_${vocab_size} \
    --groups a b e f g h i j k l m n o \
    --group-names a b e f g h i j k l m n o \
    --subsets test

fi

if [ ${stop_stage} -le 3 ]; then
  echo "Reached stop stage, quitting..."
  exit 0
fi


