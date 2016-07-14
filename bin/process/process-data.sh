#!/usr/bin/env bash

##############################################################
##                                                          ##
##     Takes a relation file where each line has format:    ##
##         e1_str \t e2_str \t rel_str \t label             ##
##                                                          ##
##############################################################

MIN_COUNT=0
MAX_SEQ=9999999
DOUBLE_VOCAB=""
CUR_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

HELP_MSG=`echo "./process-data.sh -i in_file -o out_file"$'\n' \
  "[-m min token count - below this threshold replaced with unk]"$'\n' \
  "[-p pool all relations for each entity pair] [-s max seq length to consider] "$'\n' \
  "[-c character tokens instead of whitespace delimited] [-l load_vocab] "$'\n' \
  "[-v save vocab] [-d seperate vocab for (arg1 rel arg2) and (arg2 rel arg1) triples]"$'\n' \
  "[-n normalize digits to -> #] [-g single vocab for rows and cols]"$'\n' \
  "[-b three column format] [-r reset column vocab but not rows]"$'\n' \
  "[-x convert from ep-rel pool to rel-rel pool]"
  `


PY_FILE="$CUR_DIR/StringFile2IntFile.py"
TORCH_FILE="$CUR_DIR/IntFile2Torch.lua"

while getopts i:o:v:m:l:s:pbcdrnghx opt; do
  case $opt in
  i)
      IN_FILE=$OPTARG
      PY_CMD="$PY_CMD -i $IN_FILE"
      # input supplied was a directory instead of a file
      if [[ -d $IN_FILE ]]; then
        INTERMEDIATE_FILE=`mktemp -d`
        TORCH_CMD="$TORCH_CMD -inDir ${INTERMEDIATE_FILE}"
      else
        INTERMEDIATE_FILE=`mktemp`
        TORCH_CMD="$TORCH_CMD -inFile ${INTERMEDIATE_FILE}"
      fi
      PY_CMD="$PY_CMD -o $INTERMEDIATE_FILE"
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
  g)
      PY_CMD="$PY_CMD -g"
      ;;
  p)
      TORCH_FILE="$CUR_DIR/IntFile2PoolRelationsTorch.lua"
      ;;
  b)
      PY_FILE="$CUR_DIR/BothEncoderStringFile2IntFile.py"
      TORCH_FILE="$CUR_DIR/BothEncoderIntFile2Torch.lua"
      ;;
  x)
      REL_REL_CONVERT="true"
      TORCH_FILE="$CUR_DIR/IntFile2PoolRelationsTorch.lua"
      ;;
  h)
      echo ${HELP_MSG}
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

if [[ -z "$IN_FILE" || -z "$OUT_FILE" ]]; then
  echo "Must supply input and output files"$'\n'"${HELP_MSG}"
  exit 1
fi


SAVE_VOCAB_FILE=${IN_FILE}-vocab


echo "Converting string file to int file in python"
echo "${PY_FILE} ${PY_CMD}"
python ${PY_FILE} ${PY_CMD}

echo "Converting int file to torch tensors"
echo "${TORCH_FILE} ${TORCH_CMD}"
th ${TORCH_FILE} ${TORCH_CMD}


if [[ $REL_REL_CONVERT ]]; then
    th ${CUR_DIR}/PooledEPRel2RelRel.lua -inFile ${OUT_FILE} -outFile ${OUT_FILE}-rel-rel
fi