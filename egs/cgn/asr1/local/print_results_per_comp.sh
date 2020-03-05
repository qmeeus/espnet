#!/bin/bash

components='o k l j m n g f b h a i c d'  # choose components (as space separated string)
wavdir='/users/spraak/spchdata/cgn/wav'

chmod u+x local/wer_per_component.sh

for model in exp/train_s/*; do
  echo "### ${model#*/*/} ###"
  for dir in exp/train_t/${model#*/*/} exp/train_s/${model#*/*/} ; do
    if [ -e $dir/num_jobs ]; then
      for file in $dir/grap*; do
        #echo "## file: " $file
        if [[ -d $file ]]; then
          steps/info/gmm_dir_info.pl $dir
          for x in $dir/decode*; do
            echo "## x: " $x
	    for comp in $components; do
              #echo "## Comp: " $comp
              local/wer_per_component.sh $dir $x $comp $wavdir $file
            done
          done
        fi
      done
    fi
  done
done

for model in exp/train_cleaned/*; do
  echo "### ${model#*/*/} ###"
  for dir in exp/train_cleaned/${model#*/*/}; do
    if [ -e $dir/num_jobs ]; then
      steps/info/gmm_dir_info.pl $dir
      for x in $dir/decode*; do
        for comp in $components; do
          local/wer_per_component.sh $dir $x $comp $wavdir
        done
      done
    fi
  done
done

for model in exp/nnet3_cleaned/*; do
  echo "### ${model#*/*/} ###"
  for dir in exp/nnet3_cleaned/*; do
    if [ -e $dir/num_jobs ]; then
      steps/info/gmm_dir_info.pl $dir
      for x in $dir/decode*; do
        for comp in $components; do
          local/wer_per_component.sh $dir $x $comp $wavdir
        done
      done
    fi
  done
done

for model in exp/nnet3_cleaned_1a/*; do
  echo "### ${model#*/*/} ###"
  for dir in exp/nnet3_cleaned_1a/*; do
    if [ -e $dir/num_jobs ]; then
      steps/info/gmm_dir_info.pl $dir
      for x in $dir/decode*; do
        for comp in $components; do
          local/wer_per_component.sh $dir $x $comp $wavdir
        done
      done
    fi
  done
done

for model in exp/chain_cleaned/*; do
  echo "### ${model#*/*/} ###"
  for dir in exp/chain_cleaned/*; do
    if [ -e $dir/num_jobs ]; then
      steps/info/gmm_dir_info.pl $dir
      for x in $dir/decode*; do
        for comp in $components; do
          local/wer_per_component.sh $dir $x $comp $wavdir
        done
      done
    fi
  done
done

for model in exp/chain_cleaned_1a/*; do
  echo "### ${model#*/*/} ###"
  for dir in exp/chain_cleaned_1a/*; do
    if [ -e $dir/num_jobs ]; then
      steps/info/gmm_dir_info.pl $dir
      for x in $dir/decode*; do
        for comp in $components; do
          local/wer_per_component.sh $dir $x $comp $wavdir
        done
      done
    fi
  done
done 




