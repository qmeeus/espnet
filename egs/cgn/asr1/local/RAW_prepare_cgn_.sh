#!/bin/bash 

datafiles=$(readlink -f data/datafiles.csv)
if ! [ -f "$datafiles" ]; then
echo "$datafiles not found"
exit 1
fi

if ! [ "$comp" ] || ! [ "$lang" ]; then
echo "Please select set comp/lang to include in the dataset"
exit 1
fi

if [[ $comp = *";"* ]]; then
comp="["$(echo $comp | sed "s/;/|/g")"]"
fi

if [[ $lang == *";"* ]]; then
lang=$(echo $lang | sed "s/;/|/g")
fi

local_dir=data/local/data_$tag

cat $datafiles | grep "comp-$comp" | grep "$lang" | cut -d, -f4 | sed "s#^#$cgn_root/#" > $local_dir/temptrain.flist
echo "$(wc -l temptrain.flist) files selected (see $(readlink -f $local_dir/temptrain.flist))"

local/process_flist.pl $cgn_root $train_set
# local/iso8859-to-utf8.pl $train_set.txt
iconv -f iso-8859-1 -t utf-8 $train_set.txt -o $train_set.txt
cat $train_set.utt2spk | utils/utt2spk_to_spk2utt.pl > $train_set.spk2utt

mkdir -p data/$train_set
cp $local_dir/${train_set}_wav.scp data/$train_set/wav.scp
cp $local_dir/$train_set.txt data/$train_set/text
cp $local_dir/$train_set.spk2utt data/$train_set/spk2utt
cp $local_dir/$train_set.utt2spk data/$train_set/utt2spk
cp $local_dir/$train_set.segments data/$train_set/segments
utils/filter_scp.pl data/$train_set/spk2utt $local_dir/${train_set}.spk2gender > data/$train_set/spk2gender
utils/fix_data_dir.sh data/$train_set
