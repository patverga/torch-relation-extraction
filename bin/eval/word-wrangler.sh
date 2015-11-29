#!/bin/bash

INPUT="$@"
TOP_K=6

#echo -e "\n dictionary encoder en + es \n"
#th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_dictionary-tokens.txt -model models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_dictionary/15-model -dictionary data/fixed-2013-tac/fixed-nologs/spanish/raw-data/en_es.dictionary.uniq -gpuid 1 -topK $TOP_K -input "$INPUT"

#echo -e "\n dictionary embeddings en + es \n"
#th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_dictionary-tokens.txt -embeddings models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_dictionary/15-rel-weights -dictionary data/fixed-2013-tac/fixed-nologs/spanish/raw-data/en_es.dictionary.uniq -gpuid -1 -topK $TOP_K -input "$INPUT"

#echo -e "\n dictionary encoder es only \n"
#th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_dictionary-tokens.txt -model models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_dictionary/15-model -dictionary data/fixed-2013-tac/fixed-nologs/spanish/raw-data/en_es.dictionary.uniq -gpuid 1 -topK $TOP_K -esOnly -input "$INPUT"

echo -e "\n dictionary embeddings es only \n"
th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_dictionary-tokens.txt -embeddings models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_dictionary/15-rel-weights -dictionary data/fixed-2013-tac/fixed-nologs/spanish/raw-data/en_es.dictionary.uniq -gpuid -1 -topK $TOP_K -esOnly -input "$INPUT"


#echo -e "\n no ties encoder en + es \n"
#th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_no-ties-tokens.txt -model models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_no-ties/15-model -gpuid 1 -topK $TOP_K -input "$INPUT"

#echo -e "\n no ties embeddings en + es \n"
#th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_no-ties-tokens.txt -embeddings models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_no-ties/15-rel-weights -gpuid -1 -topK $TOP_K -input "$INPUT"

#echo -e "\n no ties encoder es only \n"
#th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_no-ties-tokens.txt -model models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_no-ties/15-model -gpuid 1 -topK $TOP_K -esOnly -input "$INPUT" 

echo -e "\n no ties embeddings es only \n"
th NearestNeighbors.lua -map vocabs/no-log_min5_noramlized.en-es_no-ties-tokens.txt -embeddings models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_no-ties/15-rel-weights -gpuid -1 -topK $TOP_K -esOnly -input "$INPUT"
