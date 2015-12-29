#!/usr/bin/env bash

YEAR=$1
MODEL=$2
VOCAB=$3
GPU=$4
MAX_SEQ=$5
TUNED_PARAMS=$6
OUT=$7
EVAL_ARGS=${@:8}

TAC_EVAL_ROOT=${TH_RELEX_ROOT}/bin/tac-evaluation
source ${TAC_EVAL_ROOT}/configs/${YEAR}

# score candidate file
SCORED_CANDIDATES=`mktemp`
CAND_SCORE_CMD="th ${TH_RELEX_ROOT}/src/eval/ScoreCandidateFile.lua -candidates $CANDIDATES -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $SCORED_CANDIDATES -maxSeq $MAX_SEQ $EVAL_ARGS"
echo "Scoring candidate file: ${CAND_SCORE_CMD}"
${CAND_SCORE_CMD}

# threshold candidate file using tuned params
THRESHOLD_CANDIDATE=`mktemp`
echo "Thresholding candidate file :"
${TAC_EVAL_ROOT}/eval-scripts/threshold-scored-candidates.sh ${SCORED_CANDIDATES} ${TUNED_PARAMS} ${THRESHOLD_CANDIDATE}

# correct answers
echo "Exporting correct answers to ${OUT}_correct"
grep -Ff <(grep -Ff <(grep $'\tC\t' ${KEY} | awk '{split ($2,a,":"); print a[1]"\t"a[2]":"a[3]"\t"$4}') ${THRESHOLD_CANDIDATE} | cut -d$'\t' -f 1-8) ${CANDIDATES} > ${OUT}_correct

# wrong answers
echo "Exporting wrong answers to ${OUT}_wrong"
grep -Ff <(grep -vFf <(grep $'\tC\t' ${KEY} | awk '{split ($2,a,":"); print a[1]"\t"a[2]":"a[3]"\t"$4}') ${THRESHOLD_CANDIDATE} | cut -d$'\t' -f 1-8) ${CANDIDATES} > ${OUT}_wrong
