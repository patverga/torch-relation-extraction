#!/usr/bin/env bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
TUNED_PARAMS=$6
OUT=$7
EVAL_ARGS=${@:8}

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

# score candidate file
SCORED_CANDIDATES=`mktemp`
CAND_SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES -maxSeq $MAX_SEQ $EVAL_ARGS"
echo "Scoring candidate file: ${CAND_SCORE_CMD}"
${CAND_SCORE_CMD}

# convert scored candidate to response file
RESPONSE=`mktemp`
RESPONSE_CMD=`${TAC_ROOT}/components/bin/response.sh ${RUN_DIR}/query_expanded.xml ${SCORED_CANDIDATES} ${RESPONSE}`
echo "Converting scored candidate to response file : ${RESPONSE_CMD}"
${RESPONSE_CMD}

# post process and score response
RES_SCORE_CMD=`${TAC_EVAL_ROOT}/score-responses.sh ${YEAR} ${RESPONSE}`
echo "Evaluating response file : ${RES_SCORE_CMD}"
${RES_SCORE_CMD}
