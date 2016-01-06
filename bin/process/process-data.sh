#!/usr/bin/env bash

##############################################################
##                                                          ##
##     Takes a relation file where each line has format:    ##
##         e1_str \t e2_str \t rel_str \t label             ##
##                                                          ##
##############################################################

MIN_COUNT=0
MAX_SEQ=9999999
CHARS=""
DOUBLE_VOCAB=""
CUR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

PY_FILE="$CUR_DIR/StringFile2IntFile.py"
TORCH_FILE="$CUR_DIR/IntFile2Torch.lua"

while getopts i:o:v:m:l:s:p:cdrn opt; do
  case $opt in
  i)
      IN_FILE=$OPTARG
      PY_CMD="$PY_CMD -i $IN_FILE"
      INTERMEDIATE_FILE=`mktemp`
      PY_CMD="$PY_CMD -o $INTERMEDIATE_FILE"
      TORCH_CMD="$TORCH_CMD -inFile ${INTERMEDIATE_FILE}"
      ;;
  o)
      OUT_FILE=$OPTARG
      TORCH_CMD="$TORCH_CMD -outFile ${OUT_FILE}"
      ;;
  l)
      LOAD_VOCAB=$OPTARG
      PY_CMD="$PY_CMD -l $LOAD_VOCAB"
      ;;
  v)
      SAVE_VOCAB=$OPTARG
      PY_CMD="$PY_CMD -v $SAVE_VOCAB"
      ;;
  m)
      MIN_COUNT=$OPTARG
      PY_CMD="$PY_CMD -m $MIN_COUNT"
      TORCH_CMD="$TORCH_CMD -minCount ${MIN_COUNT}"
      ;;
  s)
      MAX_SEQ=$OPTARG
      PY_CMD="$PY_CMD -s $MAX_SEQ"
      TORCH_CMD="$TORCH_CMD -maxSeq ${MAX_SEQ}"
      ;;
  c)
      CHARS="true"
      PY_CMD="$PY_CMD -c"
      ;;
  d)
      DOUBLE_VOCAB="true"
      # this also resets the token vocab currently
      PY_CMD="$PY_CMD -d -r"
      ;;
  r)
      PY_CMD="$PY_CMD -r"
      ;;
  n)
      PY_CMD="$PY_CMD -n"
      ;;
  p)
      TORCH_FILE="IntFile2PoolRelationsTorch.lua"
      ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "$IN_FILE" || -z "$OUT_FILE" ]]
then
  echo "Not enough input args : ./process-data.sh -i IN_FILE -o OUT_FILE [-m MIN_COUNT] [-p POOL_RELATIONS] \
  [-s MAX_SEQ_LEN] [-c CHAR TOKENIZE] [-l LOAD VOCAB] [-v SAVE VOCAB] [-d DOUBLE VOCAB] [-n NUMBERS -> #]"
  exit 1
fi


SAVE_VOCAB_FILE=${IN_FILE}-vocab


echo "Converting string file to int file in python"
echo "${PY_FILE} ${PY_CMD}"
python ${PY_FILE} ${PY_CMD}

echo "Converting int file to torch tensors"
echo "${TORCH_FILE} ${TORCH_CMD}"
th ${TORCH_FILE} ${TORCH_CMD}
