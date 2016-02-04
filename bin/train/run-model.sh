#!/usr/bin/env bash

ARGS=${@:1}


RUN_CMD="th ${TH_RELEX_ROOT}/src/${MODEL}.lua
-train ${TRAIN_FILE_ROOT}/${TRAIN_FILE}
-saveModel ${SAVE}
-maxSeq $MAX_SEQ
-learningRate $LEARN_RATE
-gpuid ${GPU_ID}
-rowDim ${ROW_DIM}
-numEpochs ${MAX_EPOCHS}
-resultDir ${SAVE}
-colEncoder ${COL_ENCODER}
"

if [ "$ROW_ENCODER" ]; then
  RUN_CMD="$RUN_CMD -rowEncoder $ROW_ENCODER"
fi
if [ "$TIE_ENCODERS" ]; then
  RUN_CMD="$RUN_CMD -tieEncoders"
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

if [ "$RELATION_POOL" ]; then
  RUN_CMD="$RUN_CMD -relationPool $RELATION_POOL"
fi

if [ "$DROPOUT" ]; then
  RUN_CMD="$RUN_CMD -dropout $DROPOUT"
fi
if [ "$LAYER_DROPOUT" ]; then
  RUN_CMD="$RUN_CMD -layerDropout $LAYER_DROPOUT"
fi
if [ "$WORD_DROPOUT" ]; then
  RUN_CMD="$RUN_CMD -wordDropout $WORD_DROPOUT"
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

RUN_CMD="$RUN_CMD $ARGS"

echo "Training : ${MODEL}_${TRAIN_FILE}\tGPU : $GPU_ID"
#echo "$RUN_CMD"
${RUN_CMD} | 2>&1 tee -a "$SAVE/log"
