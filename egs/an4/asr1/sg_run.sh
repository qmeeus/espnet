#!/bin/bash -e
# Run the script

IMAGE=/users/spraak/spch/prog/spch/ESPnet/singularity.img

singularity exec --nv $IMAGE ./run.sh

