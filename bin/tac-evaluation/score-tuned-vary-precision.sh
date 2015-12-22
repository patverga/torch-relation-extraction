#!/bin/bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
TUNED_PARAMS=$6
OUT=$7
EVAL_ARGS=${@:8}

PARAM_P_BIAS="0.0 0.05 0.1 0.15"

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

SCORED_CANDIDATES=`mktemp`

# score candidate file
SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES -maxSeq $MAX_SEQ $EVAL_ARGS"
echo ${SCORE_CMD}
${SCORE_CMD}

for bias in $PARAM_P_BIAS; do
echo "Applying bias of ${bias} to tuned params and scoring"
    awk -v BIAS=$bias '{print $1 " " $2 + BIAS}' $TUNED_PARAMS > ${TUNED_PARAMS}_$bias
    EVAL_CMD="${TAC_EVAL_ROOT}/eval-scripts/evaluateScoresTunedThresh $SCORE_SCRIPT $SCORED_CANDIDATES $RUN_DIR $CONFIG $PP $KEY ${TUNED_PARAMS}_$bias ${OUT}_$bias"
    echo ${EVAL_CMD}
    ${EVAL_CMD}
done

rm ${SCORED_CANDIDATES}
