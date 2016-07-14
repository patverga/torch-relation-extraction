#!/usr/bin/env bash

FB15K_DIR=$1
PROCESSED_DIR=$2
MAX_SEQ_LEN=35
MAX_COLS=0


mkdir -p $PROCESSED_DIR

# reform the data, remove awful carriage returns
echo "Reformatting fb15k files"
cat $FB15K_DIR/text_emnlp.txt | tr -d '\r'| tr ":" " " | awk -v MAX=$MAX_SEQ_LEN -F'\t' '{split($2,tokens," "); if (length(tokens) <= MAX) print $1 "\t" $3 "\t" $2 "\t1"}' > $PROCESSED_DIR/text_emnlp.txt
cat $FB15K_DIR/train.txt | tr -d '\r'| awk -F'\t' '{print $1 "\t" $3 "\t" $2 "\t1"}' > $PROCESSED_DIR/train.txt
cat $FB15K_DIR/valid.txt | tr -d '\r'| awk -F'\t' '{print $1 "\t" $3 "\t" $2 "\t1"}' > $PROCESSED_DIR/valid.txt
cat $FB15K_DIR/test.txt | tr -d '\r'| awk -F'\t' '{print $1 "\t" $3 "\t" $2 "\t1"}' > $PROCESSED_DIR/test.txt
cat $PROCESSED_DIR/train.txt $PROCESSED_DIR/text_emnlp.txt > $PROCESSED_DIR/all-train.txt


echo "Process training data"
# this file contains entity pair - relation triples
$TH_RELEX_ROOT/bin/process/process-data.sh -i $PROCESSED_DIR/all-train.txt -o $PROCESSED_DIR/train.torch -v $PROCESSED_DIR/vocab -s $MAX_SEQ_LEN
# pooled data - maps each entity pair to all of its relations
$TH_RELEX_ROOT/bin/process/process-data.sh -i $PROCESSED_DIR/all-train.txt -o $PROCESSED_DIR/train-pooled.torch -l $PROCESSED_DIR/vocab/vocab.pkl  -p -s $MAX_SEQ_LEN
# create training data with only relations
th $TH_RELEX_ROOT/bin/process/PooledEPRel2RelRel.lua -inFile $PROCESSED_DIR/train-pooled.torch -outFile $PROCESSED_DIR/train-relations.torch -maxColumns $MAX_COLS -maxSamples $MAX_COLS


echo "Processing test data."
mkdir $PROCESSED_DIR/eval-data
th $TH_RELEX_ROOT/bin/process/util/fb15k-evaluation-process.lua -inDir $PROCESSED_DIR -outDir $PROCESSED_DIR/eval-data -vocabPrefix $PROCESSED_DIR/vocab/

# make pooled versions of valid and test data
mkdir $PROCESSED_DIR/eval-data/valid-pooled $PROCESSED_DIR/eval-data/test-pooled
th $TH_RELEX_ROOT/bin/process/PoolTestData.lua -inFile $PROCESSED_DIR/eval-data/valid/0.torch -outFile $PROCESSED_DIR/eval-data/valid-pooled/0.torch -keyFile $PROCESSED_DIR/train-pooled.torch
th $TH_RELEX_ROOT/bin/process/PoolTestData.lua -inFile $PROCESSED_DIR/eval-data/test/0.torch -outFile $PROCESSED_DIR/eval-data/test-pooled/0.torch -keyFile $PROCESSED_DIR/train-pooled.torch



### No entity pairs in training data subset ####
echo "Processing unseen entity pair data subset"
# filter entity pairs occuring in either validation or test data from the training data
grep -v -Ff <(cut -d $'\t' -f 1,2 <(cat $PROCESSED_DIR/valid.txt $PROCESSED_DIR/test.txt)) $PROCESSED_DIR/all-train.txt > $PROCESSED_DIR/filtered-train.txt

$TH_RELEX_ROOT/bin/process/process-data.sh -i $PROCESSED_DIR/filtered-train.txt -o $PROCESSED_DIR/filtered-train.torch -l $PROCESSED_DIR/vocab/vocab.pkl -s $MAX_SEQ_LEN
$TH_RELEX_ROOT/bin/process/process-data.sh -i $PROCESSED_DIR/filtered-train.txt -o $PROCESSED_DIR/filtered-train-pooled.torch -l $PROCESSED_DIR/vocab/vocab.pkl -p -s $MAX_SEQ_LEN
th $TH_RELEX_ROOT/bin/process/PooledEPRel2RelRel.lua -inFile $PROCESSED_DIR/filtered-train-pooled.torch -outFile $PROCESSED_DIR/filtered-train-relations.torch -maxColumns $MAX_COLS -maxSamples $MAX_COLS
