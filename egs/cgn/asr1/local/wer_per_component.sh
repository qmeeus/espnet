#!/bin/bash

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

modeldir=$1
decodedir=$2
comp=$3
wavdir=$4
graphdir=$5
textfile=$6

symtab=$graphdir/words.txt  # symbolic table that maps words to numbers and vice versa

# utils/best_wer returns e.g. WER 13.86 [ 4910 / 35423, 465 ins, 931 del, 3514 sub ] [PARTIAL] exp_finished/chain_cleaned/tdnn1i_sp_bi/decode_dev_s/wer_11 --> so reverse output, filter on last '_', and reverse the numbers again (can be one or two)
bestpath=$(grep WER $decodedir/wer_* | utils/best_wer.sh | rev | cut -d'_' -f -1 | rev)

if [ ! $bestpath ]
then
  bestpath=0
fi

if [ ! $bestpath -gt 0 ]
then
  exit 0;
fi

mkdir -p $decodedir/compscoring

python local/prepare_WER_comp.py $comp $bestpath $decodedir $wavdir

run.pl $decodedir/scoring/log/score.$comp.log \
    cat $decodedir/compscoring/$comp.tra \| \
    utils/int2sym.pl -f 2- $symtab \| local/filter_hyp.pl \| \
    compute-wer --text --mode=present \
     ark:$decodedir/compscoring/test_filt_$comp.txt  ark,p:- ">&" $decodedir/compscoring/wer_$comp|| exit 1;

#echo $decodedir/wer_$comp
#cat $decodedir/wer_$comp | cut -f -2

res=$(sed '2q;d' $decodedir/compscoring/wer_$comp)
echo "Component " $comp ": " $res

echo "Component " $comp ": " $res >> $textfile

exit 0;
