#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (Yusuke Fujita)
#           2021 Johns Hopkins University (Jiatong Shi)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import numpy as np
from espnet2.fileio.npy_scp import NpyScpReader
from scipy.signal import medfilt

parser = argparse.ArgumentParser(description="make rttm from decoded result")
parser.add_argument("diarize_scp")
parser.add_argument("out_rttm_file")
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--frame_shift", default=128, type=int)
parser.add_argument("--subsampling", default=1, type=int)
parser.add_argument("--median", default=1, type=int)
parser.add_argument("--sampling_rate", default=8000, type=int)
args = parser.parse_args()

scp_reader = NpyScpReader(args.diarize_scp)

with open(args.out_rttm_file, "w") as wf:
    for key in scp_reader.keys():
        data = scp_reader[key]
        a = np.where(data[:] > args.threshold, 1, 0)
        if args.median > 1:
            a = medfilt(a, (args.median, 1))
        factor = args.frame_shift * args.subsampling / args.sampling_rate
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1, 1), "constant")
            (changes,) = np.where(np.diff(frames, axis=0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                print(
                    fmt.format(
                        key,
                        s * factor,
                        (e - s) * factor,
                        key + "_" + str(spkid),
                    ),
                    file=wf,
                )

logging.info("Constructed RTTM for {}".format(args.diarize_scp))
