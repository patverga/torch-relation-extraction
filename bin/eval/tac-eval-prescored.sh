#!/bin/bash

YEAR=$1
SCORED_CANDIDATES=$2
TUNED_PARAMS=$3

source ${TH_RELEX_ROOT}/bin/tac-configs/${YEAR}

${TH_RELEX_ROOT}/bin/tac-scripts/evaluateScoresTuned $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY $TUNED_PARAMS

