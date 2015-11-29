#!/usr/bin/env bash

IN_DATA="/iesl/canvas/beroth/tac/data/merge_2013.mtx"
FAC_HOME="/home/pat/canvas/factorie"
TORCH_HOME="/home/pat/canvas/universal-schema/univSchema/torch"

# create test-mtx dir and clear out old files in it
mkdir ${TORCH_HOME}/data/test-mtx/
rm ${TORCH_HOME}/data/test-mtx/*

# export the data from factorie
${FAC_HOME}/run_class.sh cc.factorie.epistemodb.tac.ExportData --tac-data=${IN_DATA}

# get num ents and num rels
NUM_ENTS=`wc -l row-map.tsv | awk '{print $1}'`
NUM_RELS=`wc -l col-map.tsv | awk '{print $1}'`

# convert train file to torch
th ${TORCH_HOME}/process/IntTsv2Torch.lua \
-inFile train.mtx \
-outFile ${TORCH_HOME}/data/train-mtx.torch \
-addOne \
-colNames 'e1,e2,ep,rel' \
-extraNum num_rels=${NUM_RELS},num_eps=${NUM_ENTS}

# convert all the test matrices to torch
for f in `ls test-mtx/*test.mtx`; do
    th ${TORCH_HOME}/process/IntTsv2Torch.lua \
    -inFile ${f} \
    -outFile ${TORCH_HOME}/data/${f}.torch \
    -addOne \
    -colNames 'e1,e2,ep,rel,label'
done

# conver learned embeddings to torch
#th ${TORCH_HOME}/process/IntTsv2Torch.lua -inFile row.embeddings -outFile row-embeddings.torch -delim ' '
#th ${TORCH_HOME}/process/IntTsv2Torch.lua -inFile col.embeddings -outFile col-embeddings.torch -delim ' '