#!/usr/bin/env bash

GPU_ID=$1
CONFIG=$2

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/train
source ${TAC_EVAL_ROOT}/configs/${CONFIG}

DATE=`date +'%Y-%m-%d_%k'`
mkdir ${SAVE_MODEL_ROOT} ${RESULT_ROOT}

echo "Training : ${MODEL}_${TRAIN_FILE}\tGPU : $GPU_ID"
th ${TH_RELEX_ROOT}/src/${MODEL}.lua \
-loadEpEmbeddings "${TRAINED_EP}" \
-train ${TRAIN_FILE_ROOT}/${TRAIN_FILE} \
-saveModel ${SAVE_MODEL_ROOT}/${MODEL}_${TRAIN}  \
-maxSeq "$MAX_SEQ" \
-learningRate "$LEARN_RATE" \
-dropout "$DROPOUT" \
-gpuid ${GPU_ID} \
-embeddingDim "${EMBED_DIM}" \
-wordDim "${WORD_DIM}" \
-relDim "${REL_DIM}" \
-numEpochs ${MAX_EPOCHS} \
-test "$TEST_FILE" \
| tee ${RESULT_ROOT}/${MODEL}_${TRAIN}
