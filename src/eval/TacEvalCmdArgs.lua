--
-- User: pv
-- Date: 3/14/16
--

require 'torch'

local TacEvalCmdArgs = torch.class('TacEvalCmdArgs')

local cmd = torch.CmdLine()
cmd:option('-candidates', '', 'input candidate file')
cmd:option('-outFile', '', 'scored candidate out file')
cmd:option('-vocabFile', '', 'txt file containing vocab-index map')
cmd:option('-tacLabelMap', '', 'file mapping tac relations to ints for classification')
cmd:option('-dictionary', '', 'txt file containing en-es dictionary')
cmd:option('-maxSeq', 999999, 'throw away sequences longer than this')
cmd:option('-model', '', 'a trained model that will be used to score candidates')
cmd:option('-delim', ' ', 'delimiter to split lines on')
cmd:option('-threshold', -10000, 'scores will be max(threshold, score)')
cmd:option('-poolWeight', 0.0, 'weight of pooled score, weight of single score is 1-this : should be between [0,1]')
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')
cmd:option('-unkIdx', 1, 'Index to map unknown tokens to')
cmd:option('-padIdx', 2, 'Index to map pad tokens to')
cmd:option('-chars', false, 'Split tokens into characters')
cmd:option('-relations', false, 'Use full relation vectors instead of tokens')
cmd:option('-logRelations', false, 'Use log relation vectors instead of tokens')
cmd:option('-doubleVocab', false, 'double vocab so that tokens to the right of ARG1 are different then to the right of ARG2')
cmd:option('-appendEs', false, 'append @es to end of relation')
cmd:option('-normalizeDigits', '#', 'map all digits to this')
cmd:option('-tokenAppend', '', 'append this to the end of each token')
cmd:option('-fullPath', false, 'use the full input pattern without any segmenting')
cmd:option('-scoringType', 'cosine', 'How to score candidate file. cosine=uschema cosine distance, pool=pooling uschema, classifier=dist supervision classifier')
cmd:option('-scoring', 'standard', 'standard - score each relation, max - take only max per ep/tac combo, mean - avg each ep pattern to score each tac')



function TacEvalCmdArgs:parse(cmd_args)
    local params = cmd:parse(cmd_args)
    -- print the params in sorted order
    local param_array = {}
    for arg, val in pairs(params) do table.insert(param_array, arg .. ' : ' .. tostring(val)) end
    table.sort(param_array)
    for _, arg_val in ipairs(param_array) do print(arg_val) end
    return params
end