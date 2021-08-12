#!/bin/bash -e

. path.sh
. cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=8         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=8
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=CGN_train
train_sp=train_sp
train_dev=CGN_valid
recog_set="grabo patience"
# recog_set="a b e f g h i j k l m n o"

feat_sp_dir=${dumpdir}/${train_set}/nopitch; mkdir -p ${feat_sp_dir}
feat_dt_dir=${dumpdir}/${train_dev}/nopitch; mkdir -p ${feat_dt_dir}

# dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
dict=data/lang_unigram/${train_set}_${bpemode}_${nbpe}_units.txt
# bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
bpemodel=data/lang_${bpemode}/${train_set}_${bpemode}_${nbpe}
echo "dictionary: ${dict}"

lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        student_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_sp_dir}/data_${bpemode}_${nbpe}.poly.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}_${nbpe}.poly.json
fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#     echo "stage 5: Decoding"
#     if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
#            [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
#            [[ $(get_yaml.py ${train_config} etype) = transformer ]] || \
#            [[ $(get_yaml.py ${train_config} dtype) = transformer ]]; then
#         # Average ASR models
#         if ${use_valbest_average}; then
#             recog_model=model.val${n_average}.avg.best
#             opt="--log ${expdir}/results/log"
#         else
#             recog_model=model.last${n_average}.avg.best
#             opt="--log"
#         fi
#         average_checkpoints.py \
#             ${opt} \
#             --backend ${backend} \
#             --snapshots ${expdir}/results/snapshot.ep.* \
#             --out ${expdir}/results/${recog_model} \
#             --num ${n_average}

#         # Average LM models
#         if [ ${lm_n_average} -eq 0 ]; then
#             lang_model=rnnlm.model.best
#         else
#             if ${use_lm_valbest_average}; then
#                 lang_model=rnnlm.val${lm_n_average}.avg.best
#                 opt="--log ${lmexpdir}/log"
#             else
#                 lang_model=rnnlm.last${lm_n_average}.avg.best
#                 opt="--log"
#             fi
#             average_checkpoints.py \
#                 ${opt} \
#                 --backend ${backend} \
#                 --snapshots ${lmexpdir}/snapshot.ep.* \
#                 --out ${lmexpdir}/${lang_model} \
#                 --num ${lm_n_average}
#         fi
#     fi

#     pids=() # initialize pids
#     for rtask in ${recog_set}; do
#     (
#         decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
#         # feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#         # feat_recog_dir=${dumpdir}/CGN_test/nopitch/
#         feat_recog_dir=${dumpdir}/grabo_patience/nopitch/
#         # json_prefix=data_${bpemode}_${nbpe}
#         json_prefix=data
#         jsonfile=${json_prefix}.${rtask}.json

#         # split data
#         splitjson.py --parts ${nj} ${feat_recog_dir}/${jsonfile}

#         #### use CPU for decoding
#         ngpu=0

#         # set batchsize 0 to disable batch decoding
#         ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
#             asr_recog.py \
#             --config ${decode_config} \
#             --ngpu ${ngpu} \
#             --backend ${backend} \
#             --batchsize 0 \
#             --recog-json ${feat_recog_dir}/split${nj}utt/${json_prefix}.${rtask}.JOB.json \
#             --result-label ${expdir}/${decode_dir}/data.JOB.json \
#             --model ${expdir}/results/${recog_model}  \
#             --api v2

#         score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

#     ) #&
#     # pids+=($!) # store background pids
#     done
#     # i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#     # [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
#     echo "Finished"
# fi
