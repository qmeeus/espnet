#!/usr/bin/env bash -e
# TODO: This file is the same in EVERY PROJECT
#   -> Move to $PROJECT_ROOT/utils 
#   -> Call with `source utils/env.sh` $USER_DIR

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

# Add kaldi stuff to the path
export PATH=$USER_DIR/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$USER_DIR:$PATH
if ! [ -f $KALDI_ROOT/tools/config/common_path.sh ]; then
  echo "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" >&2
  exit 1
fi
. $KALDI_ROOT/tools/config/common_path.sh

# Locale and encoding for python
export LC_ALL=C
export PYTHONIOENCODING=UTF-8
export OMP_NUM_THREADS=1

# Add necessary libraries to LD_LIBRARY_PATH
if [ -d $PROJECT_ROOT/tools/chainer_ctc/ext/warp-ctc ]; then
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$PROJECT_ROOT/tools/chainer_ctc/ext/warp-ctc/build
fi

# Add utils and espnet specific binaries to the path
# TODO: shouldnt this be done before adding $USER_DIR/utils? If the user wants to overwrite some scripts...
export PATH=$PROJECT_ROOT/utils:$PROJECT_ROOT/espnet/bin:$PATH
