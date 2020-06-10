#!/bin/bash -e
# Run the script

IMAGE=/users/spraak/spch/prog/spch/ESPnet/singularity.img

if [ "$1" == "--dev" ]; then
  CVD=$2
  #if ! [[ "$CVD" =~ '^[0-9]+$' ]];  then
  #  echo "error: Not a number" >&2; exit 1
  #fi
  export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CVD
  shift; shift
fi

if [ "$#" -lt 1 ]; then
  echo "Missing command to run"
  exit 1
fi

singularity exec --nv $IMAGE $@

