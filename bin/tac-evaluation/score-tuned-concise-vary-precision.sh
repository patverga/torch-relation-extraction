#!/usr/bin/env bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
TUNED_PARAMS=$6
EVAL_ARGS=${@:7}

PARAM_P_BIAS="-0.25 -0.2 -0.15 -0.1 -.05 0.0 0.05 0.1 0.15 0.2 0.25"

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

# score candidate file
SCORED_CANDIDATES=`mktemp`
CAND_SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES -maxSeq $MAX_SEQ $EVAL_ARGS"
echo "Scoring candidate file: ${CAND_SCORE_CMD}"
${CAND_SCORE_CMD}


for bias in ${PARAM_P_BIAS}; do

    # create a p biased params file
    BIAS_PARAMS=${TUNED_PARAMS}_${bias}
    awk -v BIAS=$bias '{print $1 " " $2 + BIAS}' ${TUNED_PARAMS} > ${BIAS_PARAMS}

    # threshold candidate file using tuned params
    THRESHOLD_CANDIDATE=`mktemp`
    echo "Thresholding candidate file :"
    ${TAC_EVAL_ROOT}/eval-scripts/threshold-scored-candidates.sh ${SCORED_CANDIDATES} ${BIAS_PARAMS} ${THRESHOLD_CANDIDATE}

    # convert scored candidate to response file
    echo "Converting scored candidate to response file"
    RESPONSE=`mktemp`
    ${TAC_ROOT}/components/bin/response.sh ${RUN_DIR}/query_expanded.xml ${THRESHOLD_CANDIDATE} ${RESPONSE}

    echo "Post processing response for year $YEAR"
    RESPONSE_PP=`mktemp`
    ${TAC_EVAL_ROOT}/post-process-response.sh ${YEAR} ${PP} ${RUN_DIR} ${RESPONSE} ${RESPONSE_PP}

    echo "Evaluating response"
    echo "`${SCORE_SCRIPT} ${RESPONSE_PP} ${KEY} | grep -e F1 -e Recall -e Precision | tr '\n' '\t'`"
done