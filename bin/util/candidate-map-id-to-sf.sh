#!/usr/bin/env bash

# given a candidate file, replace SF_## Ids with canonical surface form

IN=$1
OUT=$2
SF_MAP=$3

awk -F '\t' 'BEGIN{ FS=OFS="\t" } \
NR==FNR { map[$1] = $2; next } \
NR!=FNR{ if ($1 in map) print map[$1]"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9} ' \
$SF_MAP $IN > $OUT
