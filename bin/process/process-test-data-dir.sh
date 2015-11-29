#!/usr/bin/env bash

if [ "$#" -lt 3 ]
then
  echo "Not enough input args : ./process-test-dir.sh -i IN_DIR -o OUT_DIR [-m MIN_COUNT] \
  [-s MAX_SEQ_LEN] [-c CHAR TOKENIZE] [-l LOAD VOCAB] [-v SAVE VOCAB]"
  exit 1
fi

while getopts i:o:v:m:l:s:cdrn opt; do
  case $opt in
  i)
      IN_DIR=$OPTARG
      ;;
  o)
      OUT_DIR=$OPTARG
      ;;
  l)
      LOAD_VOCAB="-l $OPTARG"
      ;;
  m)
      MIN_COUNT="-m $OPTARG"
      ;;
  s)
      MAX_SEQ="-s $OPTARG"
      ;;
  c)
      CHARS=$OPTARG
      ;;
  d)
      DOUBLE_VOCAB=$OPTARG
      ;;
  r)
      REVERSE=$OPTARG
      ;;
  n)
      NORMALIZE=$OPTARG
      ;;
  esac
done

CUR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROCESS_FILE=$CUR_DIR/process-data.sh

for f in `ls $IN_DIR | grep -v -e ints -e vocab`; do
   IN_FILE=$IN_DIR/$f
   OUT_FILE=$OUT_DIR/$f
   CMD="$PROCESS_FILE -i $IN_FILE -o $OUT_FILE $MIN_COUNT $MAX_SEQ $CHARS $LOAD_VOCAB $DOUBLE_VOCAB $NORMALIZE $REVERSE"
   echo "$CMD"
   $CMD
done