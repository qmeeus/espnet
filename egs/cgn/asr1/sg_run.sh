#!/bin/bash -e
# Run the script

IMAGE=/users/spraak/spch/prog/spch/ESPnet/singularity.img

if [ "$#" -lt 1 ]; then
  echo "Missing script to run"
  exit 1
fi

SCRIPT="$1"
shift

singularity exec --nv $IMAGE ./$SCRIPT $@

