#!/bin/bash -xe

# Preparation for CGN data by LvdW

if [ $# -lt 3 ]; then
   echo "Arguments should be <CGN root> <language> <comps> [<decodecomps>] [<train_set>] [<dev_set>]"
   echo "see ../run.sh for example."
   exit 1;
fi

cgn="$1"
lang="$2"
comps="$3"
decodecomps="${4:-$comps}"
train_set="${5:-train_s}"
dev_set="${6:-dev_s}"
tag="${7:-_s}"

base=`pwd`
dir=`pwd`/data/local/data_$tag
lmdir=`pwd`/data/local/cgn_lm_$tag
dictdir=`pwd`/data/local/dict_nosp_$tag
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh 	# Needed for KALDI_ROOT

if [ -z $SRILM ]; then
  export SRILM=$KALDI_ROOT/tools/srilm
fi
export PATH=${PATH}:$SRILM/bin/i686-m64
if ! command -v ngram-count >/dev/null 2>&1; then
  echo "$0: Error: SRILM is not available or compiled" >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_srilm.sh" >&2
  exit 1
fi

cd $dir

# create train & dev set
## Create .flist files (containing a list of all .wav files in the corpus)
rm -f tempdecode.flist
rm -f temp.flist
rm -f temptrain.flist
IFS=';'
for l in $lang; do
	for i in $decodecomps; do
        echo ${cgn}/wav/comp-${i}/${l}
		find ${cgn}/wav/comp-${i}/${l} -name '*.wav' >> tempdecode.flist
	done
	for j in $comps; do
		echo ${cgn}/wav/comp-${j}/${l}
		find ${cgn}/wav/comp-${j}/${l} -name '*.wav' >> temptrain.flist
	done
done


IFS=' '
# now split into train and dev   // -vF selects non-matching lines in file, -F the matching lines
grep -vF -f $local/comps_devset.txt temptrain.flist | grep -v 'comp-c\|comp-d' | sort >${train_set}.flist
grep -F -f $local/comps_devset.txt tempdecode.flist | grep -v 'comp-c\|comp-d' | sort >${dev_set}.flist

# create utt2spk, spk2utt, txt, segments, scp, spk2gender
for x in ${train_set} ${dev_set}; do
	mv $x.flist temp.flist
	python $local/fix_flist.py $dir $x  # deal with suffixes in wav files like _A0
	rm -f temp.flist

	$local/process_flist.pl $cgn $x

	mv ${x}_wav.scp temp.scp
	python $local/fix_wavlist.py $dir $x  # change wav.scp to point to suffix _A0 if required
	rm -f temp.scp

	# recode -d h..u8 $x.txt				# CGN is not in utf-8 by default
        # iconv -f html -t UTF-8 $x > $x.txt
	$local/iso8859-to-utf8.pl $x.txt
	cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt
done

# ### JAKOB: Recode command not present on machines, so converted html lexicon to utf-8 online and copied it to data/local/dict_nosp beforehand

# # prepare lexicon
# ## If you have a lexicon prepared, you can simply place it in $dictdir and it will be used instead of the default CGN one
# if [ -f $base/lexicon.txt ] && [ ! -f $dictdir/lexicon.txt ]; then
#         mkdir -p $dictdir
#         cp $base/lexicon.txt $dictdir
# fi

# if [ ! -f $dictdir/lexicon.txt ]; then
# 	mkdir -p $dictdir
# 	# [ -e $cgn/data/lexicon/xml/cgnlex.lex ] && cat $cgn/data/lexicon/xml/cgnlex.lex | recode -d h..u8 | perl -CSD $local/format_lexicon.pl $lang | sort >$dictdir/lexicon.txt
# 	#[ -e $cgn/data/lexicon/xml/cgnlex_2.0.lex ] && cat $cgn/data/lexicon/xml/cgnlex_2.0.lex | recode -d h..u8 | perl -CSD $local/format_lexicon.pl $lang | sort >$dictdir/lexicon.txt
# 	[ -e $cgn/data/lexicon/xml/cgnlex_2.0.lex ] && cat $cgn/data/lexicon/xml/cgnlex_2.0.lex | perl -CSD $local/format_lexicon.pl $lang | sort >$dictdir/lexicon.txt
# 	## uncomment lines below to convert to UTwente phonetic lexicon
# 	# cp $dictdir/lexicon.txt $dictdir/lexicon.orig.txt
# 	# cat $dictdir/lexicon.orig.txt | perl $local/cgn2nbest_phon.pl >$dictdir/lexicon.txt
# fi
# if ! grep -q "^<unk>" $dictdir/lexicon.txt; then
# 	echo -e "<unk>\t[SPN]" >>$dictdir/lexicon.txt
# fi
# if ! grep -q "^ggg" $dictdir/lexicon.txt; then
# 	echo -e "ggg\t[SPN]" >>$dictdir/lexicon.txt
# fi
# if ! grep -q "^xxx" $dictdir/lexicon.txt; then
# 	echo -e "xxx\t[SPN]" >>$dictdir/lexicon.txt
# fi
# # the rest
# echo SIL > $dictdir/silence_phones.txt
# echo SIL > $dictdir/optional_silence.txt
# cat $dictdir/lexicon.txt | awk -F'\t' '{print $2}' | sed 's/ /\n/g' | sort | uniq >$dictdir/nonsilence_phones.txt
# touch $dictdir/extra_questions.txt
# rm -f $dictdir/lexiconp.txt

cd $base

################ PREPARE LANGUAGE MODEL #################################
# $utils/prepare_lang.sh $dictdir "<unk>" data/local/lang_tmp_nosp data/lang_nosp

# move everything to the right place
for x in ${train_set} ${dev_set}; do
	mkdir -p data/$x
	cp $dir/${x}_wav.scp data/$x/wav.scp
	cp $dir/$x.txt data/$x/text
	cp $dir/$x.spk2utt data/$x/spk2utt
	cp $dir/$x.utt2spk data/$x/utt2spk
	cp $dir/$x.segments data/$x/segments
	$utils/filter_scp.pl data/$x/spk2utt $dir/${x}.spk2gender > data/$x/spk2gender
	$utils/fix_data_dir.sh data/$x
done

echo "Data preparation succeeded"
