#!/usr/bin/env bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
TUNED_PARAMS=$6
RESPONSE_OUT=$7
EVAL_ARGS=${@:8}

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

# score candidate file
SCORED_CANDIDATES=`mktemp`
CAND_SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES -maxSeq $MAX_SEQ $EVAL_ARGS"
echo "Scoring candidate file: ${CAND_SCORE_CMD}"
${CAND_SCORE_CMD}

# threshold candidate file using tuned params
THRESHOLD_CANDIDATE=`mktemp`
echo "Thresholding candidate file :"
${TAC_EVAL_ROOT}/eval-scripts/threshold-scored-candidates.sh ${SCORED_CANDIDATES} ${TUNED_PARAMS} ${THRESHOLD_CANDIDATE}

# convert scored candidate to response file
echo "Converting scored candidate to response file"
RESPONSE=`mktemp`
${TAC_ROOT}/components/bin/response.sh ${RUN_DIR}/query_expanded.xml ${THRESHOLD_CANDIDATE} ${RESPONSE}

echo "Post processing response for year $YEAR"
${TAC_EVAL_ROOT}/post-process-response.sh $YEAR $PP $RUN_DIR $RESPONSE $RESPONSE_OUT

echo "Evaluating response"
echo "`$SCORE_SCRIPT $RESPONSE_OUT $KEY | grep -e F1 -e Recall -e Precision | tr '\n' '\t'`"

