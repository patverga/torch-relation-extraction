#!/usr/bin/env bash

# takes a candidate file and creates a subset of the file containing lines where
# the entities are within $CUTOFF tokens apart from each other
# if LESS_THAN = false, then entities must be >= $CUTOFF tokens apart

ORIG_CANDIDATE=$1
OUT=$2
CUTOFF=$3
LESS_THAN=${4:-"true"}

if [ "$LESS_THAN" = true ]; then
    awk -v cutoff=${CUTOFF} -F'\t' '{if ($6 <= $7){diff=$7-$6} else{diff=$5-$8} if (diff < cutoff) print}' ${ORIG_CANDIDATE} > ${OUT}
else
    awk -v cutoff=${CUTOFF} -F'\t' '{if ($6 <= $7){diff=$7-$6} else{diff=$5-$8} if (diff >= cutoff) print}' ${ORIG_CANDIDATE} > ${OUT}
fi
