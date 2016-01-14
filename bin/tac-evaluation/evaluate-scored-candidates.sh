#!/bin/bash

YEAR=$1
SCORED_CANDIDATES=$2
TUNED_PARAMS=$3
OUT=$4

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

#QUERY_EXPANDED=${RUN_DIR}/query_expanded.xml

# threshold candidate file using tuned params
THRESHOLD_CANDIDATE=`mktemp`
echo "Thresholding candidate file :"
${TAC_EVAL_ROOT}/eval-scripts/threshold-scored-candidates.sh ${SCORED_CANDIDATES} ${TUNED_PARAMS} ${THRESHOLD_CANDIDATE}

# convert scored candidate to response file
echo "Converting scored candidate to response file"
RESPONSE=`mktemp`
${TAC_ROOT}/components/bin/response.sh $QUERY_EXPANDED ${THRESHOLD_CANDIDATE} ${RESPONSE}

echo "Post processing response for year $YEAR"
RESPONSE_PP=`mktemp`
${TAC_EVAL_ROOT}/post-process-response.sh $YEAR $PP $QUERY_EXPANDED $RESPONSE $RESPONSE_PP

echo "Evaluating response"
echo "`$SCORE_SCRIPT $RESPONSE_PP $KEY | grep -e F1 -e Recall -e Precision | tr '\n' '\t'`"

cp $RESPONSE $OUT