#!/bin/bash

# this script computes the total amount of data for every component (hours instead of number of wav files)

for data in data/dev_s data/dev_t data/train_s_cleaned data/train_t_cleaned; do

  # duration of wav-files
  utils/data/get_utt2dur.sh $data
  utils/data/get_reco2dur.sh $data

  # duration of utterances from segments (i.e. delete silences etc.)
  cut -d ' ' -f 1,2 $data/segments | utils/utt2spk_to_spk2utt.pl > $data/reco2utt
  awk 'FNR==NR{uttdur[$1]=$2;next}
  { for(i=2;i<=NF;i++){dur+=uttdur[$i];}
    print $1 FS dur; dur=0  }'  $data/utt2dur $data/reco2utt > $data/reco2dur_seg
done

cat data/train_s_cleaned/reco2dur data/train_t_cleaned/reco2dur > data/reco2dur_train
cat data/train_s_cleaned/reco2dur_seg data/train_t_cleaned/reco2dur_seg > data/reco2dur_seg_train
cat data/dev_s/reco2dur data/dev_t/reco2dur > data/reco2dur_dev
cat data/dev_s/reco2dur_seg data/dev_t/reco2dur_seg > data/reco2dur_seg_dev

echo ""
echo "TOTAL DURATION OF COMPONENTS IN TRAINING DATA"
python local/print_total_dur.py data train
echo ""
echo "TOTAL DURATION OF COMPONENTS IN DEV DATA"
python local/print_total_dur.py data dev
echo ""
