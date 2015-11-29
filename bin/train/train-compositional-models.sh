#!/usr/bin/env bash


# given a word model (lstm or conv) trains the various multilingual models
MODEL=$1
GPU_ID=$2
ROOT=$3

DATE=`date +'%Y-%m-%d_%k'`
SAVE_MODEL_ROOT=${ROOT}/"models/$DATE/"
RESULT_ROOT=${ROOT}/"results/$DATE/"
mkdir ${SAVE_MODEL_ROOT} ${RESULT_ROOT}
TRAINED_EP="/home/pat/canvas/universal-schema/univSchema/torch/data/fixed-2013-tac/fixed-nologs/models/uschema-filtered-reloged-50d/7-ent-weights"
TRAIN_FILE_ROOT=${ROOT}/"training-data"
TRAIN_FILES="no-log_min5_noramlized.en-es_no-ties no-log_min5_noramlized.en-es_dictionary no-log_min5_noramlized.en-es_same-spell"

for TRAIN in ${TRAIN_FILES}; do
    echo "Training : ${MODEL}_${TRAIN}\tGPU : $GPU_ID"
    th ${MODEL}.lua \
    -loadEpEmbeddings ${TRAINED_EP} \
    -train ${TRAIN_FILE_ROOT}/${TRAIN} \
    -saveModel ${SAVE_MODEL_ROOT}/${MODEL}_${TRAIN}  \
    -maxSeq 15 \
    -learningRate .0001 \
    -dropout .1 \
    -gpuid ${GPU_ID} \
    -embeddingDim 50 \
    -wordDim 100 \
    -relDim 50 \
    -numEpochs 15 \
    -test '' \
    | tee ${RESULT_ROOT}/${MODEL}_${TRAIN}
done
