#!/usr/bin/env bash

export GPU_ID=$1
export CONFIG=$2
ARGS=${@:3}


source ${TH_RELEX_ROOT}/${CONFIG}

DATE=`date +'%Y-%m-%d_%H'`
export LOG_ROOT="${LOG_ROOT}/$DATE/"
mkdir -p ${LOG_ROOT}
cp ${TH_RELEX_ROOT}/${CONFIG} $SAVE

source ${TH_RELEX_ROOT}/bin/train/gen-run-cmd.sh
RUN_CMD="$RUN_CMD $ARGS"

echo "Training : ${MODEL}_${TRAIN_FILE}\tGPU : $GPU_ID"
echo "$RUN_CMD"
${RUN_CMD} | 2>&1 tee -a "$LOG_ROOT/log"