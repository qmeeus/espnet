#!/bin/bash -e

outdir=$1/results
shift

if [ -d "$outdir" ]; then
  snapshot=$(ls -halt $outdir/snapshot.ep.* 2> /dev/null | head -n1 | rev | cut -d" " -f1 | rev)
  if [ -f "$outdir/log" ]; then
    previous_index=$(ls $outdir/log.* | xargs -I{} basename {} | sort -nr | head -n +2 | cut -d. -f2 2>/dev/null)
    [ "$previous_index" ] && i=$(($previous_index + 1)) || i=1
    cp $outdir/log $outdir/log.$i
  fi
fi

! [ -z $snapshot ] && cmd="$@ --resume $snapshot" || cmd="$@"

echo "$cmd"
bash $cmd

# vim: ts=2 sw=2 et :
