#!/usr/bin/env bash

#########################################################
##
##  script for hyperparameter tuning using grid search
##  first arg must be config file
##  accepts comma seperated lists of args
##
#########################################################

CONFIG=$1
shift

source ${TH_RELEX_ROOT}/${CONFIG}
LEARN_RATE_ARGS=$LEARN_RATE
LAYERS_ARGS=$LAYERS
DROPOUT_ARGS=$DROPOUT
WORD_DIM_ARGS=$WORD_DIM
REL_DIM_ARGS=$REL_DIM
EMBED_DIM_ARGS=$EMBED_DIM

export MAX_EPOCHS=5

while [[ $# > 1 ]]
do
    key="$1"
    case $key in
            -g|--gpuid|--gpu)
            export GPU_ID="$2"
            shift # past argument
        ;;
            -l|--learning-rate)
            LEARN_RATE_ARGS=`echo "$2" | sed 's/,/ /g'`
            shift # past argument
        ;;
            -y|--layers)
            LAYERS_ARGS=`echo "$2" | sed 's/,/ /g'`
            shift # past argument
        ;;
            -d|--dropout)
            DROPOUT_ARGS=`echo "$2" | sed 's/,/ /g'`
            shift # past argument
        ;;
            -w|--worddim|--word-dim)
            WORD_DIM_ARGS=`echo "$2" | sed 's/,/ /g'`
            shift # past argument
        ;;
            -r|--reldim|--rel-dim)
            REL_DIM_ARGS=`echo "$2" | sed 's/,/ /g'`
            shift # past argument
        ;;
            -e|--embeddim|--embed-dim)
            EMBED_DIM_ARGS=`echo "$2" | sed 's/,/ /g'`
            shift # past argument
        ;;
            *)  # unknown option
        ;;
    esac
    shift # past argument or value
done

echo $LEARN_RATE_ARGS

for LEARN_RATE in ${LEARN_RATE_ARGS}; do
for LAYERS in ${LAYERS_ARGS}; do
for DROPOUT in ${DROPOUT_ARGS}; do
for WORD_DIM in ${WORD_DIM_ARGS}; do
for REL_DIM in ${REL_DIM_ARGS}; do
for EMBED_DIM in ${EMBED_DIM_ARGS}; do

    export $LEARN_RATE $LAYERS $DROPOUT $WORD_DIM $REL_DIM $EMBED_DIM
    PARAMS="learnrate-${LEARN_RATE}_layers-${LAYERS}_dropout-${DROPOUT}_worddim-${WORD_DIM}_reldim-${REL_DIM}_embeddim-${EMBED_DIM}"
    echo $PARAMS

    export DATE=`date +'%Y-%m-%d_%k'`
    export SAVE="${SAVE_MODEL_ROOT}/tuning/${PARAMS}__${DATE}"
    mkdir -p ${SAVE}
    # save a copy of the config in the model dir
    cp ${TH_RELEX_ROOT}/${CONFIG} $SAVE
    ${TH_RELEX_ROOT}/bin/train/run-model.sh

done;done;done;done;done;done
