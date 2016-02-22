#!/usr/bin/env bash

export GPU_ID=$1
export CONFIG=$2
ARGS=${@:3}


source ${TH_RELEX_ROOT}/${CONFIG}

DATE=`date +'%Y-%m-%d_%H'`
export LOG_ROOT="${LOG_ROOT}/$DATE/"
mkdir -p ${LOG_ROOT}
cp ${TH_RELEX_ROOT}/${CONFIG} $LOG_ROOT

source ${TH_RELEX_ROOT}/bin/train/gen-run-cmd.sh
RUN_CMD="$RUN_CMD $ARGS"

echo "Training : ${MODEL}_${TRAIN_FILE}\tGPU : $GPU_ID"
if [ "$GPU_ID" -ge 0 ]; then
    RUN_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID $RUN_CMD -gpuid 0"
fi
echo "$RUN_CMD"
eval ${RUN_CMD}| 2>&1 tee -a "$LOG_ROOT/log"