#!/bin/bash

TUNE_YEAR=$1
TEST_YEAR=$2
MODEL=$3
VOCAB=$4
GPU=$5
MAX_SEQ=$6
OUT=$7
EVAL_ARGS=${@:8}

# tune thresholds
${TH_RELEX_ROOT}/bin/tac-evaluation/tac-eval-tune-thresh.sh $TUNE_YEAR $MODEL $VOCAB $GPU $MAX_SEQ $OUT $EVAL_ARGS

# use tuned thresholds and evaluate on test year
${TH_RELEX_ROOT}/bin/tac-evaluation/tac-eval-score-tuned.sh $TEST_YEAR $MODEL $VOCAB $GPU $MAX_SEQ $OUT/params $OUT $EVAL_ARGS