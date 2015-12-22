#!/bin/bash
response=$1
key=$2

# Generate slotslist if not provided.
if [ $# -lt 3 ]
then
   slotlist=`mktemp`
   cut -f1,2 $response \
   | tr '\t' ':' \
   | sort -u \
   > $slotlist
else
   slotlist=$3
fi

java -cp /iesl/canvas/beroth/workspace/tackbp2014/eval/2014/eval_2014 SFScore $response $key  nocase anydoc | grep -P '\tRecall:|\tPrecision:|\tF1:'

# Delete generated slotlist.
if [ $# -lt 3 ]
then
   rm $slotlist
fi


