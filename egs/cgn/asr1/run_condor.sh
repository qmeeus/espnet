#!/usr/bin/bash -e
# Usage: ./run_condor.sh JOB_FILE [--rm] [--interactive] [OPTIONS]

if [ $# -lt 1 ] || ! [ -f "$1" ]; then
  echo Missing job file
  exit 1
fi

JOB_FILE="$1.tmp"
cp "$1" $JOB_FILE
shift

while [[ "$1" == -* ]]; do
  case $1 in
    --rm|-r)
      CLEAN=true;;
    --interactive|-i)
      INTERACTIVE=-interactive;;
    *)
      echo "Invalid option $1" && exit 1;;
  esac
  shift
done

while [[ $# -gt 0 ]]; do
  named_arg=$1
  if [[ $named_arg =~ ^[a-Z]+=.*$ ]]; then
    NAME=$(echo $named_arg | cut -d= -f1)
    ARG="$(echo $named_arg | cut -d= -f2)"
    sed -i "s#\$$NAME#$ARG#g" $JOB_FILE
  elif [ "$named_arg" ]; then
    echo "Unprocessed argument: $named_arg"
  fi
  shift
done

if [ "$INTERACTIVE" ]; then
  sed -i "/^NiceUser.*/d" $JOB_FILE
  sed -Ei "s:^(\+RequestWallTime\s+\=)\s+[0-9]+:\1 14000:g" $JOB_FILE
fi

condor_submit $INTERACTIVE $JOB_FILE

condor_q

if [ "$CLEAN" ]; then
  rm $JOB_FILE
fi
