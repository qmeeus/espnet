#!/usr/bin/bash -e

. setup.sh
. path.sh

dataset="data/CGN_ALL_"
logdir="exp/make_fbank/CGN_ALL_"
fbankdir="fbank_"

# Requirements:
# 1. ${dataset} exists and contains wav.scp
#   $ head -1 $dataset/wav.scp
#   fv400086 sox -t wav /users/spraak/spchdata/cgn/wav/comp-a/vl/fv400086.wav -b 16 -t wav - remix - |
# 2. ${dataset} contains utt2spk and spk2utt
#   $ head -n1 $dataset/utt2spk
#   V40170-fv400170.73 V40170
#   $ head -n1 $dataset/spk2utt
#   V40170 V40170-fv400170.73 V40170-fv400170.74 V40170-fv400170.75 V40170-fv400170.76
# 3. ${dataset} contains segments
#   $ head -n1 $dataset/segments
#   V40170-fv400170.73 fv400170 203.052 206.586

local/make_fbank_pitch.sh --nj 8 --write_utt2num_frames true "$dataset" "$logdir" "$fbankdir"
