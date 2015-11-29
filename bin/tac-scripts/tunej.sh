#!/bin/bash
# tunej.sh <key_file> <response_prefix> <params>*
# This script takes files of the form <response_prefix><param> and optimizes
# f-score on the TAC collection.
# Two files are written out:
# <response_prefix>tuned -- the response being the result of tuning
# <response_prefix>params -- the optimal parameter value for each relation

key=$1
SCORE_SCRIPT=$2
RESPONSE_PREFIX=$3
OUT=$4
PARAMS=${@:5}
mkdir $OUT

RELLIST=`$TAC_ROOT/bin/get_expand_config.sh rellist $TAC_ROOT/config/rellist`
RESPONSE=$OUT/tuned
#RESPONSE=$3tuned
BEST_PARAMS_FILE=$OUT/params
#BEST_PARAMS_FILE=$3params
cp ${RESPONSE_PREFIX}$5 $RESPONSE

slotlist=`mktemp`
cut -f1,2 $RESPONSE \
| tr '\t' ':' \
| sort -u \
> $slotlist

for i in 1 2
do
echo iteration $i
> ${BEST_PARAMS_FILE}

while read REL; do
  echo $REL
  OLD_FSCORE=0.0
  USED_JPARAM=$5
  TMP_RESPONSE=`mktemp`
  for JPARAM in ${PARAMS}
  do
    echo 'j = '${JPARAM}
    grep -v $REL $RESPONSE > $TMP_RESPONSE
    grep $REL ${RESPONSE_PREFIX}${JPARAM} >> $TMP_RESPONSE
    echo ${RESPONSE_PREFIX}${JPARAM} 
    wc -l $TMP_RESPONSE
    echo "$SCORE_SCRIPT $TMP_RESPONSE $key slots=$slotlist nocase anydoc"
    FSCORE=`$SCORE_SCRIPT $TMP_RESPONSE $key slots=$slotlist nocase anydoc | grep 'F1: ' | sed 's#.*F1:\(.*\)#\1#'` # | sed 's/^[ \t]*//;s/[ \t]*$//'`
    if [[ "$FSCORE" == "NaN" ]]
    then
        FSCORE=0.0
    fi
    #echo $'\r'"F1: $FSCORE"
    if [[ `echo "$FSCORE > $OLD_FSCORE" | bc` == 1 ]]; 
    then
     echo $'\r'"F1: $FSCORE"
     OLD_FSCORE=$FSCORE
     USED_JPARAM=$JPARAM
    fi
  done
  grep -v $REL $RESPONSE > $TMP_RESPONSE
  grep $REL ${RESPONSE_PREFIX}${USED_JPARAM} >> $TMP_RESPONSE
  mv $TMP_RESPONSE $RESPONSE

  echo $REL $USED_JPARAM >> ${BEST_PARAMS_FILE}
done < $RELLIST
done

rm $slotlist
