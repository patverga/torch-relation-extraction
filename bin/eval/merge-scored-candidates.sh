#!/bin/bash
#
# takes 2 scored candidate files and combines there scores, 
# file 2's scores are weighted by WEIGHT

SCORED_CANDIDATE_1=$1
SCORED_CANDIDATE_2=$2
COMBINED=$3
WEIGHT=$4

awk -v weight=$WEIGHT -F'\t' 'NR==FNR{ a[$1$2$3$4$5$6$7$8]=$9;next } NR!=FNR{ if ($1$2$3$4$5$6$7$8 in a){s=a[$1$2$3$4$5$6$7$8]; print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"s+$9*weight; } else print $0 }' $SCORED_CANDIDATE_1 $SCORED_CANDIDATE_2 > $COMBINED

awk -F'\t' 'NR==FNR{ a[$1$2$3$4$5$6$7$8]=$9;next } NR!=FNR{ if ($1$2$3$4$5$6$7$8 in a){} else print $0}' $COMBINED $SCORED_CANDIDATE_1 > leftovers

cat leftovers >> $COMBINED

rm leftovers
