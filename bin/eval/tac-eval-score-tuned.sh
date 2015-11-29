#!/bin/bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
OTHER_ARGS=$6
TUNED_PARAMS=$7
OUT=$8

source /home/pat/canvas/universal-schema/univSchema/torch/bin/tac-configs/${YEAR}


SCORED_CANDIDATES=${CANDIDATES}-scored-$RANDOM

# score candidate file
SCORE_CMD="th /home/pat/canvas/universal-schema/univSchema/torch/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES  -maxSeq $MAX_SEQ $OTHER_ARGS"
echo $SCORE_CMD
$SCORE_CMD

#/home/pat/canvas/universal-schema/univSchema/torch/david-evaluation/evaluateScoresTuned $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $OUT $TUNED_PARAMS| grep -e F1 -e Tuning
/home/pat/canvas/universal-schema/univSchema/torch/david-evaluation/evaluateScoresTunedThresh $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $TUNED_PARAMS $OUT

rm $SCORED_CANDIDATES
