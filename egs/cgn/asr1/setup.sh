#!/bin/bash -e

#export CUDA_HOME=/users/spraak/spch/prog/spch/cuda-10.0
#CUDNN_PATH=/users/spraak/spch/prog/spch/cudnn-7.6

#export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_PATH/lib64:/.singularity.d/libs
#export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
#export CUDA_PATH=$CUDA_HOME

CONDA_HOME=/users/spraak/qmeeus/spchdisk/bin/anaconda3
CONDA_INIT=$CONDA_HOME/etc/profile.d/conda.sh
CONDA_ENV=espnet2
source $CONDA_INIT
echo "Activate conda environment $CONDA_ENV"
conda activate $CONDA_ENV
