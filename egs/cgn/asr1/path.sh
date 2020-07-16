#!/bin/bash -e

export CUDA_HOME=/users/spraak/spch/prog/spch/cuda-10.0
CUDNN_PATH=/users/spraak/spch/prog/spch/cudnn-7.6

export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_PATH/lib64:/.singularity.d/libs
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export CUDA_PATH=$CUDA_HOME

PROJECT_ROOT=/users/spraak/spch/prog/spch/ESPnet
KALDI_ROOT=$PROJECT_ROOT/tools/kaldi

# Setup kaldi environment
if [ -f $KALDI_ROOT/tools/env.sh ]; then
  . $KALDI_ROOT/tools/env.sh
fi

# Add kaldi stuff to the path
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
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
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$PROJECT_ROOT/tools/chainer_ctc/ext/warp-ctc/build

# Setup environment
if [ -e $PROJECT_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $PROJECT_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $PROJECT_ROOT/tools/venv/bin/activate
fi

# Add utils and espnet specific binaries to the path
export PATH=$PROJECT_ROOT/utils:$PROJECT_ROOT/espnet/bin:$PATH

