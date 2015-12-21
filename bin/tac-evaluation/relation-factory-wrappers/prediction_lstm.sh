#!/usr/bin/env bash

IN=$1
OUT=$2

MODEL=/home/pat/canvas/torch-relation-extraction/models/lstm-bi-maxpool-paper/2015-12-20_10/4-model
VOCAB=/home/pat/canvas/torch-relation-extraction/vocabs/no-log_min5_noramlized.en-tokens.txt
GPU=0
EVAL_ARGS=""
MAX_SEQ=20

th /home/pat/canvas/torch-relation-extraction/src/eval/ScoreCandidateFile.lua -candidates $IN -vocabFile $VOCAB -model $MODEL -gpuid $GPU -threshold 0 -outFile $OUT -maxSeq $MAX_SEQ $EVAL_ARGS