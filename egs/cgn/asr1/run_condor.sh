#!/usr/bin/bash -e

if [ $# -lt 1 ] || ! [ -f "$1" ]; then
  echo Missing job file
  exit 1
fi

if [ "$2" == "--rm" ]; then
  CLEAN=true
  shift
fi

JOB_FILE="$1.tmp"
cp "$1" $JOB_FILE
shift

for named_arg in $@; do
  NAME=$(echo $named_arg | cut -d= -f1)
  ARG=$(echo $named_arg | cut -d= -f2)
  sed -i "s/\$$NAME/$ARG/g" $JOB_FILE
done

condor_submit $JOB_FILE

condor_q

if [ "$CLEAN" ]; then
  rm $JOB_FILE
fi
