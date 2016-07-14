#!/usr/bin/env bash

# file formats:
#   FB_TYPE_FILE = 2 col tsv file mapping fb_id entities to their fb type
#      ie. '/m/010039	/location/statistical_region'
#
#   TEXT_MENTIONS = 3 col tsv file mapping fb_id entities to mentions
#      ie. '/m/0thvy	In fact , cabinet specialists in $MENTION_START Somerset $MENTION_END , Kentucky can help you find the best cabinets around at the best price .	  1'
#
#   TEXT_RELATIONS = 3 col tsv file mapping fb_id entities to text relations
#       ie. '/m/0thvy	doctor	  1'

OUT_DIR=$1
FB_TYPE_FILE=$2
TEXT_MENTIONS=$3
TEXT_RELATIONS=$4

# params for train data
MAX_SEQ_LEN=20
MIN_TOKEN_COUNT=25
MAX_COLS=0
# params for test data
VALID_PORTION=.2
TEST_PORTION=.2
NEGS_PER_POS=100
HOLD_OUT_ENTITIES=14000
# fill for column 1 for compatibility with 4 col format
ENTITY_PREFIX="____"


echo "Splitting freebase types into train, valid, and test portions"
python ${TH_RELEX_ROOT}/bin/process/entity-data/SeperateFbTestData.py -i ${FB_TYPE_FILE} -o ${OUT_DIR}/fb-data -p ${ENTITY_PREFIX} -n ${NEGS_PER_POS} -v ${VALID_PORTION} -t ${TEST_PORTION} -l ${HOLD_OUT_ENTITIES}


echo "Combining freebase and text training data"
cp ${OUT_DIR}/fb-data/fb_train.mtx ${OUT_DIR}/all_train.mtx
if [ "$TEXT_MENTIONS" ]; then
    awk -F'\t' -v PREFIX=${ENTITY_PREFIX} '{print PREFIX"\t"$0}' ${TEXT_MENTIONS} >> ${OUT_DIR}/all_train.mtx
fi
if [ "$TEXT_RELATIONS" ]; then
    awk -F'\t' -v PREFIX=${ENTITY_PREFIX} '{print PREFIX"\t"$0}' ${TEXT_RELATIONS} >> ${OUT_DIR}/all_train.mtx
fi


# create filtered version of training set to test unseen entities
grep -v -Ff ${OUT_DIR}/fb-data/heldout-entities.txt ${OUT_DIR}/all_train.mtx > ${OUT_DIR}/all_train_filtered.mtx


echo "Processing training data"
# this file contains entity pair - relation triples
./bin/process/process-data.sh -i ${OUT_DIR}/all_train.mtx -o ${OUT_DIR}/train.torch -v ${OUT_DIR}/vocab -s ${MAX_SEQ_LEN} -m ${MIN_TOKEN_COUNT}
./bin/process/process-data.sh -i ${OUT_DIR}/all_train_filtered.mtx -o ${OUT_DIR}/train_filtered.torch -l ${OUT_DIR}/vocab/vocab.pkl -s ${MAX_SEQ_LEN} -m ${MIN_TOKEN_COUNT}

# pooled data - maps each entity pair to all of its relations
./bin/process/process-data.sh -i ${OUT_DIR}/all_train.mtx -o ${OUT_DIR}/train-pooled.torch -l ${OUT_DIR}/vocab/vocab.pkl -s ${MAX_SEQ_LEN} -m ${MIN_TOKEN_COUNT} -p
./bin/process/process-data.sh -i ${OUT_DIR}/all_train_filtered.mtx -o ${OUT_DIR}/train_filtered-pooled.torch -l ${OUT_DIR}/vocab/vocab.pkl -s ${MAX_SEQ_LEN} -m ${MIN_TOKEN_COUNT} -p

# create training data with only relations
th ${TH_RELEX_ROOT}/bin/process/PooledEPRel2RelRel.lua -inFile ${OUT_DIR}/train-pooled.torch -outFile ${OUT_DIR}/train-relations.torch -maxColumns ${MAX_COLS} -maxSamples ${MAX_COLS}
th ${TH_RELEX_ROOT}/bin/process/PooledEPRel2RelRel.lua -inFile ${OUT_DIR}/train_filtered-pooled.torch -outFile ${OUT_DIR}/train_filtered-relations.torch -maxColumns ${MAX_COLS} -maxSamples ${MAX_COLS}

echo "Processing test data"
mkdir -p ${OUT_DIR}/valid/{seen,unseen,seen-pooled,unseen-pooled} ${OUT_DIR}/test/{seen,unseen,seen-pooled,unseen-pooled}
${TH_RELEX_ROOT}/bin/process/process-data.sh -i ${OUT_DIR}/fb-data/valid_unseen -o ${OUT_DIR}/valid/unseen -l ${OUT_DIR}/vocab/vocab.pkl
${TH_RELEX_ROOT}/bin/process/process-data.sh -i ${OUT_DIR}/fb-data/valid_seen -o ${OUT_DIR}/valid/seen -l ${OUT_DIR}/vocab/vocab.pkl
${TH_RELEX_ROOT}/bin/process/process-data.sh -i ${OUT_DIR}/fb-data/test_unseen -o ${OUT_DIR}/test/unseen -l ${OUT_DIR}/vocab/vocab.pkl
${TH_RELEX_ROOT}/bin/process/process-data.sh -i ${OUT_DIR}/fb-data/test_seen -o ${OUT_DIR}/test/seen -l ${OUT_DIR}/vocab/vocab.pkl

# create pooled versions of valid and test files for rowless models
th ${TH_RELEX_ROOT}/bin/process/PoolTestData.lua -inDir ${OUT_DIR}/valid/unseen -outFile ${OUT_DIR}/valid/unseen-pooled -keyFile ${OUT_DIR}/train-pooled.torch;
th ${TH_RELEX_ROOT}/bin/process/PoolTestData.lua -inDir ${OUT_DIR}/valid/seen -outFile ${OUT_DIR}/valid/seen-pooled -keyFile ${OUT_DIR}/train-pooled.torch;
th ${TH_RELEX_ROOT}/bin/process/PoolTestData.lua -inDir ${OUT_DIR}/test/unseen -outFile ${OUT_DIR}/test/unseen-pooled -keyFile ${OUT_DIR}/train-pooled.torch;
th ${TH_RELEX_ROOT}/bin/process/PoolTestData.lua -inDir ${OUT_DIR}/test/seen -outFile ${OUT_DIR}/test/seen-pooled -keyFile ${OUT_DIR}/train-pooled.torch;
