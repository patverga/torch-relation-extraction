--
-- User: pat
-- Date: 9/17/15
--

package.path = package.path .. ";src/?.lua;src/nn-modules/?.lua"

require 'torch'
require 'rnn'
require 'ViewTable'
require 'ReplicateAs'
require 'SelectLast'
require 'VariableLengthJoinTable'
require 'VariableLengthConcatTable'
require 'NoUpdateLookupTable'
require 'NoUnReverseBiSequencer'
require 'WordDropout'
require 'EncoderPool'

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
cmd:option('-chars', false, 'Split tokens into characters')
cmd:option('-relations', false, 'Use full relation vectors instead of tokens')
cmd:option('-logRelations', false, 'Use log relation vectors instead of tokens')
cmd:option('-doubleVocab', false, 'double vocab so that tokens to the right of ARG1 are different then to the right of ARG2')
cmd:option('-appendEs', false, 'append @es to end of relation')
cmd:option('-normalizeDigits', true, 'map all digits to #')
cmd:option('-tokenAppend', '', 'append this to the end of each token')
cmd:option('-fullPath', false, 'use the full input pattern without any segmenting')


local params = cmd:parse(arg)
local function to_cuda(x) return params.gpuid >= 0 and x:cuda() or x end
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local in_vocab = 0
local out_vocab = 0


--- convert sentence to tac tensor using tokens ---
local function token_tensor(arg1_first, pattern_rel, vocab_map, dictionary, start_idx, end_idx, use_full_pattern)
    local idx = 0
    local token_ids = {}
    local tokens = {}

    local first_arg = arg1_first and '$ARG1' or '$ARG2'
    local second_arg = arg1_first and '$ARG2' or '$ARG1'
    if params.tokenAppend ~= '' then
        first_arg = first_arg .. params.tokenAppend
        second_arg = second_arg .. params.tokenAppend
    end

    for token in string.gmatch(pattern_rel, "[^" .. params.delim .. "]+") do
        if dictionary[token] then token = dictionary[token]
        elseif params.tokenAppend ~= '' then token = token .. params.tokenAppend
        end
        if (idx >= start_idx and idx < end_idx) or use_full_pattern then
            if params.chars then
                for c in token:gmatch"." do
                    table.insert(tokens, c)
                end
                table.insert(tokens, ' ')
            else
                table.insert(tokens, token)
            end
        end
        idx = idx + 1
    end

    if not use_full_pattern then
        table.insert(token_ids, vocab_map[first_arg] or params.unkIdx)
        if params.chars then table.insert(token_ids, vocab_map[' '] or params.unkIdx) end
    end

    for i = 1, #tokens do
        local token
        if not params.logRelations or #tokens <= 4 or i <= 2 or i > #tokens -2 then
            token = tokens[i]
        elseif i == 3 then
            token = '[' .. math.floor(torch.log(#tokens - 4)/torch.log(2)) .. ']'
        end
        if token then
            if params.doubleVocab then token = token .. '_' .. (arg1_first and '$ARG1' or '$ARG2') end
            local id = vocab_map[token] or params.unkIdx
            table.insert(token_ids, id)
            if id == params.unkIdx then out_vocab = out_vocab + 1 else in_vocab = in_vocab + 1 end
        end
    end

    if not use_full_pattern then table.insert(token_ids, vocab_map[second_arg] or params.unkIdx) end
    local pattern_tensor = torch.Tensor(token_ids)
    return pattern_tensor, #tokens
end

--- convert sentence to tac tensor using whole relation tensor ---
local function rel_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx, use_full_pattern)
    local rel_string
    if not use_full_pattern then
        local idx = 0
        local tokens = {}
        for token in string.gmatch(pattern_rel, "[^" .. params.delim .. "]+") do
            if params.tokenAppend ~= '' then token = token .. params.tokenAppend end
            if idx >= start_idx and idx < end_idx then table.insert(tokens, token) end
            idx = idx + 1
        end

        local first_arg = arg1_first and '$ARG1' or '$ARG2'
        if params.tokenAppend ~= '' then first_arg = first_arg .. params.tokenAppend end
        rel_string = first_arg .. ' '
        for i = 1, #tokens do
            if not params.logRelations or #tokens <= 4 or i <= 2 or i > #tokens - 2 then
                rel_string = rel_string .. tokens[i] .. ' '
            elseif i == 3 then
                rel_string = rel_string .. '[' .. math.floor(torch.log(#tokens - 4) / torch.log(2)) .. ']' .. ' '
            end
        end
        local second_arg = arg1_first and '$ARG2' or '$ARG1'
        if params.tokenAppend ~= '' then second_arg = second_arg .. params.tokenAppend end
        rel_string = rel_string .. second_arg
    else
        rel_string = pattern_rel
    end
    if params.appendEs then rel_string = rel_string .. "@es" end
    local id = -1
    local len = 0
    if vocab_map[rel_string] then
        id = vocab_map[rel_string]
        len = 1
        in_vocab = in_vocab + 1
    else
        out_vocab = out_vocab + 1
    end
    local pattern_tensor = torch.Tensor({id})
    return pattern_tensor, len
end

--- process a single line from a candidate file ---
local function process_line(line, vocab_map, dictionary)
    local query_id, tac_rel, sf_2, doc_info, start_1, end_1, start_2, end_2, pattern_rel
    = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")

    if params.normalizeDigits and not params.fullPath then pattern_rel = pattern_rel:gsub("%d", "") end

    local tac_tensor = torch.Tensor({vocab_map[tac_rel] or params.unkIdx})

    -- we only want tokens between the two args
    local start_idx = tonumber(end_1)
    local end_idx = tonumber(start_2)
    local arg1_first = true
    if (start_idx > end_idx) then
        start_idx, end_idx, arg1_first = tonumber(end_2), tonumber(start_1), false
    end

    local pattern_tensor, seq_len
    if params.relations then
        pattern_tensor, seq_len = rel_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx, params.fullPath)
    else
        pattern_tensor, seq_len = token_tensor(arg1_first, pattern_rel, vocab_map, dictionary, start_idx, end_idx, params.fullPath)
    end

    pattern_tensor = pattern_tensor:view(1, pattern_tensor:size(1))
    tac_tensor = tac_tensor:view(1, tac_tensor:size(1))
    local out_line = query_id .. '\t' .. tac_rel .. '\t' .. sf_2 .. '\t' .. doc_info .. '\t'
            .. start_1 .. '\t' .. end_1 .. '\t' .. start_2 .. '\t' .. end_2 .. '\t'

    return out_line, pattern_tensor, tac_tensor, seq_len
end

--- process the candidate file and convert to torch ---
local function process_file(vocab_map, dictionary)
    local line_num = 0
    local max_seq = 0
    local data = {}
    print('Processing data')
    for line in io.lines(params.candidates) do
        local out_line, pattern_tensor, tac_tensor, seq_len = process_line(line, vocab_map, dictionary)
        max_seq = math.max(seq_len, max_seq)
        if not data[seq_len] then data[seq_len] = {out_line={}, pattern_tensor={}, tac_tensor={}} end
        local seq_len_data = data[seq_len]
        table.insert(seq_len_data.out_line, out_line)
        table.insert(seq_len_data.pattern_tensor, pattern_tensor)
        table.insert(seq_len_data.tac_tensor, tac_tensor)
        line_num = line_num + 1
        if line_num % 10000 == 0 then io.write('\rline : ' .. line_num); io.flush() end
    end
    print ('\rProcessed ' .. line_num .. ' lines')
    return data, max_seq
end

-- TODO this only works for uschema right now
local function score_tac_relation(text_encoder, kb_rel_table, pattern_tensor, tac_tensor)
    if torch.type(text_encoder) == 'nn.EncoderPool' then pattern_tensor = pattern_tensor:view(pattern_tensor:size(1), 1, pattern_tensor:size(2)) end
    if torch.type(kb_rel_table) == 'nn.EncoderPool' then tac_tensor = tac_tensor:view(tac_tensor:size(1), 1, tac_tensor:size(2)) end

    local tac_encoded = kb_rel_table:forward(to_cuda(tac_tensor)):clone()
    local pattern_encoded = text_encoder:forward(to_cuda(pattern_tensor)):clone()

    if tac_encoded:dim() == 3 then tac_encoded = tac_encoded:view(tac_encoded:size(1), tac_encoded:size(3)) end
    if pattern_encoded:dim() == 3 then pattern_encoded = pattern_encoded:view(pattern_encoded:size(1), pattern_encoded:size(3)) end
    local x = { tac_encoded, pattern_encoded }

    local score = to_cuda(nn.CosineDistance())(x):double()
    --    local score = to_cuda(nn.Sum(2))(to_cuda(nn.CMulTable())(x)):double()
    return score
end

--- score the data returned by process_file ---
local function score_data(data, max_seq, text_encoder, kb_rel_table)
    print('Scoring data')
    -- open output file to write scored candidates file
    local out_file = io.open(params.outFile, "w")
    for seq_len = 1, math.min(max_seq, params.maxSeq) do
        if data[seq_len] then
            io.write('\rseq length : ' .. seq_len); io.flush()
            local seq_len_data = data[seq_len]
            --- batch
--            local start = 1
--            while start <= #seq_len_data do
            local pattern_tensor = nn.JoinTable(1)(seq_len_data.pattern_tensor)
            local tac_tensor = nn.JoinTable(1)(seq_len_data.tac_tensor)
            local scores = score_tac_relation(text_encoder, kb_rel_table, pattern_tensor, tac_tensor)
            local out_lines = seq_len_data.out_line
            for i = 1, #out_lines do
                local score = math.max(params.threshold, scores[i])
                out_file:write(out_lines[i] .. score .. '\n')
            end
        end
    end
    out_file:close()
end

local function load_maps()
    local vocab_map = {}
    for line in io.lines(params.vocabFile) do
        local token, id = string.match(line, "([^\t]+)\t([^\t]+)")
        if token and id then
            id = tonumber(id)
            if id > 1 then vocab_map[token] = id end
        end
    end
    local dictionary = {}
    if params.dictionary ~= '' then
        for line in io.lines(params.dictionary) do
            -- space seperated
            local en, es = string.match(line, "([^\t]+) ([^\t]+)")
            dictionary[es] = en
        end
    end
    return vocab_map, dictionary
end



---- main

-- process the candidate file
local data, max_seq = process_file(load_maps())

-- load model
local model = torch.load(params.model)
local kb_rel_table = to_cuda(model.kb_rel_table ~= nil and model.kb_rel_table or model.encoder)
local text_encoder = to_cuda(model.text_encoder ~= nil and model.text_encoder or model.encoder)
kb_rel_table:evaluate();text_encoder:evaluate()

-- score and export candidate file
score_data(data, max_seq, text_encoder, kb_rel_table)
print ('\nDone, found ' .. in_vocab .. ' in vocab tokens and ' .. out_vocab .. ' out of vocab tokens.')
