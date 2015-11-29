#!/bin/bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
OTHER_ARGS=$6
OUT=`pwd`/$7

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

SCORED_CANDIDATES=`mktemp`

# score candidate file
SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES  -maxSeq $MAX_SEQ $OTHER_ARGS"
echo $SCORE_CMD
$SCORE_CMD

${TH_RELEX_ROOT}/eval-scripts/evaluateScoresTopTuneThresh $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $OUT | grep -e F1 -e Tuning

rm $SCORED_CANDIDATES
