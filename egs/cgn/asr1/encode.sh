#!/bin/bash

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Usage: encode.sh $DATASET $MODEL_CKPT $OUTPUT_FILE"
fi

source init_conda
conda activate espnet-stable

. path.sh

decode_config=conf/encode.yaml
jsonfile=${1:-dump/grabo_patience/nopitch/data.grabo.json}
recog_model=${2:-exp/CGN_train_pytorch_sti_transformer_lr10.0_ag8_p.5_specaug/results/model.val5.avg.best}
output_dir=$(realpath $(dirname $recog_model)/../encode)
output_file=${3:-predictions}

echo Snapshot: $recog_model
if [ -d $output_dir/$output_file ]; then
  printf "\033[0;31m$output_dir/${output_file}.h5 exists. Continue? y/N\033[0m " && read yesno
  case $yesno in
    y|Y) : ;;
    *) exit 1 ;;
  esac
fi

dataset=$(basename $jsonfile | cut -d. -f2)
outdir=$output_dir/$dataset
mkdir -p $outdir
sti_encode.py \
    --config $decode_config \
    --ngpu 1 \
    --backend pytorch \
    --batchsize 4 \
    --recog-json $jsonfile \
    --outdir $outdir \
    --outfile $output_file \
    --model $recog_model
