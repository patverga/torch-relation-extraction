#!/usr/bin/env bash

export GPU_ID=$1
CONFIG=$2

source ${TH_RELEX_ROOT}/${CONFIG}

export DATE=`date +'%Y-%m-%d_%k'`
export SAVE="${SAVE_MODEL_ROOT}/$DATE"
mkdir -p ${SAVE}
# save a copy of the config in the model dir
cp ${TH_RELEX_ROOT}/${CONFIG} $SAVE


${TH_RELEX_ROOT}/bin/train/run-model.sh