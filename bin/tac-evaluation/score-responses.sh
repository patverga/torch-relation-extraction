#!/bin/bash

YEAR=$1
RESPONSES=${@:2}

source ${TH_RELEX_ROOT}/bin/tac-evaluation/configs/${YEAR}


echo "Merging responses"
if [ "$#" -gt 2 ]; then
  MERGED_RESPONSE=`mktemp`
  $TAC_ROOT/components/bin/merge_responses.sh $RUN_DIR/query_expanded.xml $RESPONSES > $MERGED_RESPONSE
else
  MERGED_RESPONSE=$2
fi

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
${TAC_EVAL_ROOT}/post-process-response.sh $YEAR $PP $RUN_DIR $MERGED_RESPONSE $RESPONSE_PP


echo "Scoring merged response"
echo "`$SCORE_SCRIPT $RESPONSE_PP $KEY | grep -e F1 -e Recall -e Precision | tr '\n' '\t'`"

