#!/usr/bin/env bash

RUN_CMD="th ${TH_RELEX_ROOT}/src/${MODEL}.lua -train ${TRAIN_FILE_ROOT}/${TRAIN_FILE} -maxSeq $MAX_SEQ -learningRate $LEARN_RATE -rowDim ${ROW_DIM} -numEpochs ${MAX_EPOCHS} -resultDir ${LOG_ROOT} -colEncoder ${COL_ENCODER}"

if [ "$SAVE_MODEL" ]; then
  RUN_CMD="$RUN_CMD -saveModel $LOG_ROOT"
fi
if [ "$GPU_ID" ]; then
  RUN_CMD="$RUN_CMD -gpuid $GPU_ID"
fi


if [ "$ROW_ENCODER" ]; then
  RUN_CMD="$RUN_CMD -rowEncoder $ROW_ENCODER"
fi
if [ "$TIE_ENCODERS" ]; then
  RUN_CMD="$RUN_CMD -tieEncoders"
fi
if [ "$SHARED_VOCAB" ]; then
  RUN_CMD="$RUN_CMD -sharedVocab"
fi
if [ "$TOKEN_DIM" ]; then
  RUN_CMD="$RUN_CMD -tokenDim $TOKEN_DIM"
fi
if [ "$COL_DIM" ]; then
  RUN_CMD="$RUN_CMD -colDim $COL_DIM"
fi
if [ "$BI_DIRECTIONAL" ]; then
  RUN_CMD="$RUN_CMD -bi $BI_DIRECTIONAL"
fi
if [ "$LAYERS" ]; then
  RUN_CMD="$RUN_CMD -layers $LAYERS"
fi
if [ "$POOL_LAYER" ]; then
  RUN_CMD="$RUN_CMD -poolLayer $POOL_LAYER"
fi
if [ "$NON_LINEAR_LAYER" ]; then
  RUN_CMD="$RUN_CMD -nonLinearLayer $NON_LINEAR_LAYER"
fi
if [ "$MODEL_TYPE" ]; then
  RUN_CMD="$RUN_CMD -modelType $MODEL_TYPE"
fi
if [ "$COMPOSITIONAL" ]; then
  RUN_CMD="$RUN_CMD -compositional"
fi
if [ "$CRITERION" ]; then
  RUN_CMD="$RUN_CMD -criterion $CRITERION"
fi
if [ "$MARGIN" ]; then
  RUN_CMD="$RUN_CMD -margin $MARGIN"
fi

if [ "$RELATION_POOL" ]; then
  RUN_CMD="$RUN_CMD -relationPool $RELATION_POOL"
fi
if [ "$K" ]; then
  RUN_CMD="$RUN_CMD -k $K"
fi

if [ "$DROPOUT" ]; then
  RUN_CMD="$RUN_CMD -dropout $DROPOUT"
fi
if [ "$L2" ]; then
  RUN_CMD="$RUN_CMD -l2Reg $L2"
fi
if [ "$EPSILON" ]; then
  RUN_CMD="$RUN_CMD -epsilon $EPSILON"
fi
if [ "$LAYER_DROPOUT" ]; then
  RUN_CMD="$RUN_CMD -layerDropout $LAYER_DROPOUT"
fi
if [ "$WORD_DROPOUT" ]; then
  RUN_CMD="$RUN_CMD -wordDropout $WORD_DROPOUT"
fi

if [ "$COL_NORM" ]; then
  RUN_CMD="$RUN_CMD -colNorm $COL_NORM"
fi
if [ "$ROW_NORM" ]; then
  RUN_CMD="$RUN_CMD -rowNorm $ROW_NORM"
fi
if [ "$CLIP_GRADS" ]; then
  RUN_CMD="$RUN_CMD -clipGrads $CLIP_GRADS"
fi

if [ "$TEST_FILE" ]; then
  RUN_CMD="$RUN_CMD -test $TEST_FILE"
fi
if [ "$TAC_YEAR" ]; then
  RUN_CMD="$RUN_CMD -tacYear $TAC_YEAR"
fi
if [ "$TRAINED_EP" ]; then
  RUN_CMD="$RUN_CMD -loadRowEmbeddings $TRAINED_EP"
fi
if [ "$TRAINED_REL" ]; then
  RUN_CMD="$RUN_CMD -loadColEmbeddings $TRAINED_REL"
fi
if [ "$LOAD_MODEL" ]; then
  RUN_CMD="$RUN_CMD -loadModel $LOAD_MODEL"
fi
if [ "$LOAD_ENCODER" ]; then
  RUN_CMD="$RUN_CMD -loadEncoder $LOAD_ENCODER"
fi
if [ "$VOCAB" ]; then
  RUN_CMD="$RUN_CMD -vocab $VOCAB"
fi
if [ "$EVAL_ARGS" ]; then
  RUN_CMD="$RUN_CMD -evalArgs $EVAL_ARGS"
fi

if [ "$EVAL_FREQ" ]; then
  RUN_CMD="$RUN_CMD -evaluateFrequency $EVAL_FREQ"
fi
if [ "$BATCH_SIZE" ]; then
  RUN_CMD="$RUN_CMD -batchSize $BATCH_SIZE"
fi

export RUN_CMD
