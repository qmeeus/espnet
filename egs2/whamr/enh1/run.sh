#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set=tr_mix_single_reverb_min_8k
valid_set=cv_mix_single_reverb_min_8k
test_sets="tt_mix_single_reverb_min_8k"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --audio_format wav \
    --spk_num 1 \
    --ngpu 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_beamformer_mvdr_wpe.yaml \
    --use_dereverb_ref true \
    --use_noise_ref true \
    --inference_model "valid.loss.best.pth" \
    "$@"

#    --enh_config ./conf/tuning/train_enh_beamformer_wpd_magloss.yaml \
#    --enh_config ./conf/tuning/train_enh_beamformer_wpd.yaml \
#    --enh_config ./conf/tuning/train_enh_beamformer_mvdr.yaml \
#    --enh_config ./conf/tuning/train_enh_beamformer_mvdr_wpe.yaml \
#    --enh_config ./conf/tuning/train_enh_beamformer_mvdr_nara_wpe.yaml \