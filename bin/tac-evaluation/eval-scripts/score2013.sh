#!/bin/bash
# provenance invariant scorer for 2013 data
response=$1
key=$2
#optargs="${@:3}"

echo java -cp /iesl/canvas/beroth/tac/relationfactory/evaluation/bin/ SFScore $response $key anydoc
java -cp /iesl/canvas/beroth/tac/relationfactory/evaluation/bin/ SFScore $response $key anydoc | grep -P '\tRecall:|\tPrecision:|\tF1:'

