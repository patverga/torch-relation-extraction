--
-- User: pat
-- Date: 9/17/15
--

package.path = package.path .. ";src/?.lua;src/nn-modules/?.lua;src/eval/?.lua"

--[[
Takes a tac candidate file, tab seperated vocab idx file, and a trained uschema encoder model
and exports a scored candidtate file to outfile
]]--
local cmd = torch.CmdLine()
cmd:option('-candidates', '', 'input candidate file')
cmd:option('-outFile', '', 'scored candidate out file')
cmd:option('-vocabFile', '', 'txt file containing vocab-index map')
cmd:option('-dictionary', '', 'txt file containing en-es dictionary')
cmd:option('-maxSeq', 999999, 'throw away sequences longer than this')
cmd:option('-model', '', 'a trained model that will be used to score candidates')
cmd:option('-delim', ' ', 'delimiter to split lines on')
cmd:option('-threshold', 0, 'scores will be max(threshold, score)')
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')
cmd:option('-unkIdx', 1, 'Index to map unknown tokens to')
cmd:option('-padIdx', 2, 'Index to map unknown tokens to')
cmd:option('-chars', false, 'Split tokens into characters')
cmd:option('-relations', false, 'Use full relation vectors instead of tokens')
cmd:option('-logRelations', false, 'Use log relation vectors instead of tokens')
cmd:option('-doubleVocab', false, 'double vocab so that tokens to the right of ARG1 are different then to the right of ARG2')
cmd:option('-appendEs', false, 'append @es to end of relation')
cmd:option('-normalizeDigits', true, 'map all digits to #')
cmd:option('-tokenAppend', '', 'append this to the end of each token')
cmd:option('-fullPath', false, 'use the full input pattern without any segmenting')
cmd:option('-pool', false, 'pool all relations containing each entity pair to make decicions')

local params = cmd:parse(arg)

---- main
require 'SentenceClassifier'
require 'PoolClassifier'
local scorer = params.pool and PoolClassifier(params) or SentenceClassifier(params)
-- process the candidate file
local data, max_seq = scorer:process_file(scorer:load_maps())
-- score and export candidate file
scorer:score_data(data, max_seq)
print ('\nDone, found ' .. scorer.in_vocab .. ' in vocab tokens and ' .. scorer.out_vocab .. ' out of vocab tokens.')
