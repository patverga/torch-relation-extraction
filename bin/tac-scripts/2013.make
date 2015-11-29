# Makefile to generate TAC-response. 
# Pipeline starts with query.xml and generates response.
#
# This produces a 2013 run merging the output from old and new modules.
#
# Author: Benjamin Roth

# copies query.xml from location specified in config file
query.xml:
	cp $(shell $(TAC_ROOT)/bin/get_expand_config.sh query.xml) $@

# Adds expansiond to original queries and explicitly lists relations.
query_expanded.xml: query.xml
	$(TAC_ROOT)/components/bin/expand_query.sh $+ $@

# Retrieves ranked list document ids/files.
dscore: query_expanded.xml
	$(TAC_ROOT)/components/bin/retrieve.sh $+ $@

# Tokenizes/splits sentences from retrieved docs.
drank: query_expanded.xml dscore
	$(TAC_ROOT)/components/bin/split_sentences2.sh $+ $@

# Tags sentences.
dtag: drank
	$(TAC_ROOT)/components/bin/tagging.sh $+ $@

# Creates title org mappings
title_org.tabs: dtag
	$(TAC_ROOT)/components/bin/title_org_extract.sh $+ $@

# Candidates from sentences where Query string and tags match.
candidates: query_expanded.xml dtag dscore
	$(TAC_ROOT)/components/bin/candidates2013.sh $+ $@

# Converts candidates into protocol-buffer format.
candidates.pb: candidates
	$(TAC_ROOT)/components/bin/cands_to_proto.sh $+ $@

# Runs candidates through dependency parser so they can be used by the dependency matcher.
candidates.parsed.pb:	candidates.pb
	$(TAC_ROOT)/components/bin/parse_candidates.sh $+ $@

# Extracts features on a per-sentence level.
sfeatures: candidates.pb
	$(TAC_ROOT)/components/bin/sfeatures.sh $+ $@

# Predicts slot fillers on a per-sentence level.
predictions_classifier: sfeatures
	echo run the scala code
#$(TAC_ROOT)/components/bin/predictions.sh $+ $@

# Generates TAC-response with 'lsv' team id.
response_classifier: query_expanded.xml predictions_classifier
	$(TAC_ROOT)/components/bin/response.sh $+ $@

# Generates TAC-response with 'lsv' team id.
response_classifier_david: query_expanded.xml predictions_classifier_david
	$(TAC_ROOT)/components/bin/response.sh $+ $@

# Response from pattern matches.
response_patterns: query_expanded.xml candidates.pb
	$(TAC_ROOT)/components/bin/pattern_response.sh $+ $@

# Response from dependency pattern matches.
response_dependency_patterns: query_expanded.xml candidates.parsed.pb
	$(TAC_ROOT)/components/bin/dependency_pattern_response.sh $+ $@

# Response from dependency pattern matches.
response_cuny_patterns: query_expanded.xml candidates
	$(TAC_ROOT)/components/bin/cuny_pattern_response.sh $+ $@

# Response from induced patterns.
response_induced_patterns: query_expanded.xml candidates
	$(TAC_ROOT)/components/bin/induced_pattern_response.sh $+ $@

# Response from matching query name expansions.
response_alternate_names: query_expanded.xml dtag dscore
	$(TAC_ROOT)/components/bin/alternate_names.sh $+ $@

# Returns answer tuples from Wikipedia retrieval.
wiki_slots: query.xml
	$(TAC_ROOT)/components/bin/wiki_slots.sh $+ $@

# Matches answer tuples from Wikipedia against candidates.
response_wiki: query_expanded.xml wiki_slots candidates
	$(TAC_ROOT)/components/bin/list_match_response.sh $+ $@

## Disallowed by 2013 guidelines.
## Returns answer tuples from querying freebase.
#freebase_slots: query_expanded.xml
#	$(TAC_ROOT)/components/bin/freebase_slots.sh $+ $@
## Matches answer tuples from freebase against candidates.
#freebase_response: query_expanded.xml freebase_slots candidates
#	$(TAC_ROOT)/components/bin/list_match_response.sh $+ $@

# modules that run fast (1)
response_fast: query_expanded.xml response_alternate_names response_classifier response_induced_patterns response_patterns
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@

# modules that run fast (1)
response_fast_david: query_expanded.xml response_alternate_names response_classifier_david response_classifier #response_induced_patterns response_patterns
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@

# modules that run fast (1)
response_fast_david_nosvm: query_expanded.xml response_alternate_names response_classifier_david response_induced_patterns response_patterns
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@


# Modules that have precision >~40% (2)
response_prec: query_expanded.xml response_alternate_names response_dependency_patterns response_induced_patterns response_patterns
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@

# Modules that do not include manual patterns. Wiki response is also excluded, as it uses manual patterns, too.
response_nomanual: query_expanded.xml response_alternate_names response_classifier response_induced_patterns
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@

# Modules without parsing (= all but PRIS patterns). (5)
response_nosyntax: query_expanded.xml response_alternate_names response_classifier response_induced_patterns response_patterns response_wiki
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@

# All modules. (3)
response_all: query_expanded.xml response_alternate_names response_classifier response_dependency_patterns response_induced_patterns response_patterns response_wiki
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@

# All modules. (3)
response_all_david: query_expanded.xml response_alternate_names response_classifier response_classifier_david response_dependency_patterns response_induced_patterns response_patterns response_wiki
	$(TAC_ROOT)/components/bin/merge_responses.sh $+ > $@


# With additional per:employee_or_member_of slots from orgs associated with predicted titles. (4)
response_all_orgs_from_titles: query_expanded.xml candidates response_all title_org.tabs
	$(TAC_ROOT)/components/bin/orgs_from_titles.sh $+ > $@

response: response_all_pp13
	cp -v $< $@

response2012: response_all_pp12
	cp -v $< $@

# Template for postprocessing responses for 2013 format
%_pp13: % query_expanded.xml /dev/null
	$(TAC_ROOT)/components/bin/postprocess2013_nolinkstat.sh $+ $@

# Postprocess response for 2012 format
%_pp12: % query_expanded.xml 
	$(TAC_ROOT)/components/bin/postprocess.sh $+ $@


# If Mintz feature set is used, things get a little more complicated.
# TODO: needs testing.
# Postagging
%.ptag.pb: %.pb
	$(TAC_ROOT)/components/bin/experimental/pos_tag.sh $+ $@

# Extracts features on a per-sentence level.
sfeatures_mintz: candidates.parsed.ptag.pb
	$(TAC_ROOT)/components/bin/sfeatures.sh $+ $@

# Predicts slot fillers on a per-sentence level.
predictions_mintz: sfeatures_mintz
	$(TAC_ROOT)/components/bin/predictions.sh $+ $@

# Generates TAC-response with 'lsv' team id.
response_mintz: query_expanded.xml predictions_mintz
	$(TAC_ROOT)/components/bin/response.sh $+ $@	

# Parsing
#%.parse.pb: %.pb
#	$TAC_ROOT/components/bin/parse_candidates.sh $+ $@


### 2013 submission runs ###

VALIDATION2013_ROOT=$(TAC_ROOT)/evaluation/eval2013/validation

define validate2013
	$(VALIDATION2013_ROOT)/check_kbp_slot-filling.pl $(VALIDATION2013_ROOT)/doc_ids_english.txt query.xml $@
endef

define createRun2013
	$(TAC_ROOT)/bin/addRunId.sh $(1) $(2) > $@
	$(validate2013)
endef

run2013_fast:	response_fast_pp13
	$(call createRun2013,$<,lsv1)

run2013_prec:	response_prec_pp13
	$(call createRun2013,$<,lsv2)

run2013_all:	response_all_pp13
	$(call createRun2013,$<,lsv3)
	
run2013_recall:	response_all_orgs_from_titles_pp13
	$(call createRun2013,$<,lsv4)

#run2013_nomanual:	response_nomanual_pp13
#	$(call createRun2013,$<,lsv5)

run2013_nosyntax:	response_nosyntax_pp13
	$(call createRun2013,$<,lsv5)

2013: run2013_fast run2013_prec run2013_all run2013_recall run2013_nosyntax

