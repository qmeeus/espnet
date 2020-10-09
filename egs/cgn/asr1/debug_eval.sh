
. init_conda
conda activate espnet

tag=transformer_12_6_2048_512_8_a.4_do.1
traindir=exp/train_unigram_5000_mono/$tag/train
model=$traindir/results/model.acc.best
#outdir=$traindir/evaluate
outdir=exp/debug/evaluate

./evaluate.sh --ngpu 0 \
    --target wordpiece --vocab-size 5000 \
    --decode-config conf/.bak/decode_v2_ctcweight0.5.yaml \
    --tag $tag \
    --output_dir $outdir \
    --recog-model $model

