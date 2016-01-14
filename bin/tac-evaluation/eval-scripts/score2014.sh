#!/bin/bash
response=$1
key=$2

java -cp ${TH_RELEX_ROOT}/bin/tac-evaluation/eval-scripts/SFScore-2014 SFScore $response $key  nocase anydoc | grep -P '\tRecall:|\tPrecision:|\tF1:'
