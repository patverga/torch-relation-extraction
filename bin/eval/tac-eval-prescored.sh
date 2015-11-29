#!/bin/bash

YEAR=$1
SCORED_CANDIDATES=$2
TUNED_PARAMS=$3

source /home/pat/canvas/universal-schema/univSchema/torch/bin/tac-configs/${YEAR}


#/home/pat/canvas/universal-schema/univSchema/torch/david-evaluation/evaluateScoresTuned $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $OUT $TUNED_PARAMS| grep -e F1 -e Tuning
/home/pat/canvas/universal-schema/univSchema/torch/david-evaluation/evaluateScoresTuned $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $TUNED_PARAMS

