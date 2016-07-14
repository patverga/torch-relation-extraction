#!/usr/bin/env bash


# grab the data and unpack
wget -O $TH_RELEX_ROOT/data/entity_type_data.tar.gz https://people.cs.umass.edu/~pat/data/entity_type_data.tar.gz
tar -xvzf $TH_RELEX_ROOT/data/entity_type_data.tar.gz

$TH_RELEX_ROOT/bin/process/entity-data/process-entity-data.sh $TH_RELEX_ROOT/data/entity-type-data/processed_data $TH_RELEX_ROOT/data/entity-type-data/entity_fb-types $TH_RELEX_ROOT/data/entity-type-data/text_relations.mtx
