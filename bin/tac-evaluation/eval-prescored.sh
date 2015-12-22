#!/bin/bash

YEAR=$1
SCORED_CANDIDATES=$2
TUNED_PARAMS=$3

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

${TAC_EVAL_ROOT}/eval-scripts/evaluateScoresTunedThresh $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $TUNED_PARAMS

