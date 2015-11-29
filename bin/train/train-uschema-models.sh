#!/usr/bin/env bash


GPU_ID=$1
MODEL="UniversalSchema"
DATE=`date +'%Y-%m-%d_%k'`
SAVE_MODEL_ROOT="${TH_RELEX_ROOT}/models/$DATE/"
RESULT_ROOT="${TH_RELEX_ROOT}/results/$DATE/"
mkdir ${SAVE_MODEL_ROOT} ${RESULT_ROOT}
#TRAINED_EP="data/fixed-2013-tac/fixed-nologs/models/uschema-filtered-reloged-50d/7-ent-weights"
TRAIN_FILE_ROOT="${TH_RELEX_ROOT}/training-data"
TRAIN_FILES="relogged_min5_noramlized.en relogged_min5_noramlized.en-es_es-appended"

for TRAIN in ${TRAIN_FILES}; do
    echo "Training : ${MODEL}_${TRAIN}\tGPU : $GPU_ID"
    th ${TH_RELEX_ROOT}/src/${MODEL}.lua \
    -train ${TRAIN_FILE_ROOT}/${TRAIN} \
    -saveModel ${SAVE_MODEL_ROOT}/${MODEL}_${TRAIN}  \
    -maxSeq 15 \
    -batchSize 1024 \
    -learningRate .001 \
    -dropout .1 \
    -gpuid ${GPU_ID} \
    -embeddingDim 50 \
    -wordDim 100 \
    -relDim 50 \
    -numEpochs 15 \
    -test '' \
    | tee ${RESULT_ROOT}/${MODEL}_${TRAIN}
done