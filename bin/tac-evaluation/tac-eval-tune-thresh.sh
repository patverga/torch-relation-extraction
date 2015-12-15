#!/bin/bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
OUT=$6
EVAL_ARGS=${@:7}


TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

SCORED_CANDIDATES=`mktemp`

# score candidate file
SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES -maxSeq $MAX_SEQ $EVAL_ARGS"
echo ${SCORE_CMD}
${SCORE_CMD}

TUNE_CMD="${TAC_EVAL_ROOT}/eval-scripts/evaluateScoresTuneThresh $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $OUT"
echo ${TUNE_CMD}
${TUNE_CMD} | grep -e F1 -e Tuning

rm ${SCORED_CANDIDATES}