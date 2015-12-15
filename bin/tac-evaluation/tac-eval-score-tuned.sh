#!/bin/bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
TUNED_PARAMS=$6
OUT=$7
EVAL_ARGS=$8


TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

SCORED_CANDIDATES=`mktemp`

# score candidate file
SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES -maxSeq $MAX_SEQ $EVAL_ARGS"
echo ${SCORE_CMD}
${SCORE_CMD}

EVAL_CMD="${TAC_EVAL_ROOT}/eval-scripts/evaluateScoresTunedThresh $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $TUNED_PARAMS $OUT"
echo ${EVAL_CMD}
${EVAL_CMD}

rm ${SCORED_CANDIDATES}
