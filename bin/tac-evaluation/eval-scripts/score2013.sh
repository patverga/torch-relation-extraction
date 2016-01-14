#!/bin/bash
# provenance invariant scorer for 2013 data
response=$1
key=$2
#optargs="${@:3}"

java -cp ${TH_RELEX_ROOT}/bin/tac-evaluation/eval-scripts/SFScore-2013 SFScore $response $key anydoc | grep -P '\tRecall:|\tPrecision:|\tF1:'

