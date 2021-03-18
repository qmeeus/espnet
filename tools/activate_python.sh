#!/bin/bash -e

CONDA_HOME=/users/spraak/qmeeus/spchtemp/bin/anaconda3
CONDA_INIT=$CONDA_HOME/etc/profile.d/conda.sh
CONDA_ENV=espnet2
source $CONDA_INIT
# echo "Activate conda environment $CONDA_ENV"
conda activate $CONDA_ENV
