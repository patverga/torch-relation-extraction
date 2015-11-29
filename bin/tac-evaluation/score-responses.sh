#!/bin/bash

YEAR=$1
RESPONSES=${@:2}

source tac-configs/${YEAR}


echo "Merging responses"
if [ "$#" -gt 2 ]; then
  MERGED_RESPONSE=`mktemp`
  $TAC_ROOT/components/bin/merge_responses.sh $RUN_DIR/query_expanded.xml $RESPONSES > $MERGED_RESPONSE
else
  MERGED_RESPONSE=$2
fi

echo "Post processing for year $YEAR"
RESPONSE_PP=`mktemp`
if [[ $PP == "pp13" ]]; then
  $TAC_ROOT/components/bin/postprocess2013.sh $MERGED_RESPONSE $RUN_DIR/query_expanded.xml /dev/null $RESPONSE_PP
elif [[ $PP == "pp12" ]]; then
 # $TAC_ROOT/components/bin/postprocess2012.sh $MERGED_RESPONSE $RUN_DIR/query_expanded.xml /dev/null $RESPONSE_PP
  $TAC_ROOT/components/bin/postprocess2013-12.sh $MERGED_RESPONSE $RUN_DIR/query_expanded.xml /dev/null $RESPONSE_PP
else
  cat $MERGED_RESPONSE > $RESPONSE_PP
fi  

echo "Scoring merged response"
echo "`$SCORE_SCRIPT $RESPONSE_PP $KEY | grep -e F1 -e Recall -e Precision | tr '\n' '\t'`"

