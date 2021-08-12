#!/bin/bash -e

outdir=$1/results
shift

if [ -d "$outdir" ]; then
  snapshot=$(ls -halt $outdir/snapshot.ep.* 2> /dev/null | head -n1 | rev | cut -d" " -f1 | rev)
fi

if ! [ -z $snapshot ]; then
  cmd="$@ --resume $snapshot"
  cp $outdir/log $outdir/log_$(date +"%Y%m%d_%H%M")
else
  cmd="$@"
fi

echo "$cmd"
bash $cmd

# vim: ts=2 sw=2 et :
