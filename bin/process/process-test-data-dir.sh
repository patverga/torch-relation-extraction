#!/usr/bin/env bash

if [ "$#" -lt 3 ]
then
  echo "Not enough input args : ./process-test-dir.sh IN_DIR OUT_DIR VOCAB_FROM_TRAINING_DATA"
  exit 1
fi

IN_DIR=$1
OUT_DIR=$2
VOCAB=$3
ARGS=${@:4}

CUR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
PROCESS_FILE=$CUR_DIR/process-data.sh

for f in `ls $IN_DIR `; do
   IN_FILE=$IN_DIR/$f
   OUT_FILE=$OUT_DIR/$f
   CMD="$PROCESS_FILE -i $IN_FILE -o $OUT_FILE -l $VOCAB $ARGS"
   echo "$CMD"
   $CMD
done