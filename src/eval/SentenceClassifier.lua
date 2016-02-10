--
-- User: pat
-- Date: 2/5/16
--

package.path = package.path .. ";src/?.lua;src/nn-modules/?.lua;src/eval/?.lua"

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

grad = require 'autograd'
grad.optimize(true) -- global

local expand_as = function(input)
    local target_tensor = input[1]
    local orig_tensor = input[2]
    local expanded_tensor = torch.expand(orig_tensor, target_tensor:size())
    return expanded_tensor
end

local SentenceClassifier = torch.class('SentenceClassifier')

function SentenceClassifier:__init(params)
    self.params = params
    if self.params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(self.params.gpuid + 1) else require 'nn' end

    -- load model
    local model = torch.load(self.params.model)
    self.kb_encoder = self:to_cuda(model.kb_col_table and model.kb_col_table or (model.row_encoder and model.row_encoder or (model.col_encoder and model.col_encoder or model.encoder)))
    self.text_encoder = self:to_cuda(model.text_encoder and model.text_encoder or (model.col_encoder and model.col_encoder or model.encoder))
    self.net = self:to_cuda(model.net)
    self.net:evaluate(); self.kb_encoder:evaluate(); self.text_encoder:evaluate()

    self.in_vocab = 0
    self.out_vocab = 0
end

function SentenceClassifier:to_cuda(x) return self.params.gpuid >= 0 and x:cuda() or x end



--- convert sentence to tac tensor using tokens ---
function SentenceClassifier:token_tensor(arg1_first, pattern_rel, vocab_map, dictionary, start_idx, end_idx, use_full_pattern)
    local idx = 0
    local token_ids = {}
    local tokens = {}

    local first_arg = arg1_first and '$ARG1' or '$ARG2'
    local second_arg = arg1_first and '$ARG2' or '$ARG1'
    if self.params.tokenAppend ~= '' then
        first_arg = first_arg .. self.params.tokenAppend
        second_arg = second_arg .. self.params.tokenAppend
    end

    for token in string.gmatch(pattern_rel, "[^" .. self.params.delim .. "]+") do
        if dictionary[token] then token = dictionary[token]
        elseif self.params.tokenAppend ~= '' then token = token .. self.params.tokenAppend
        end
        if (idx >= start_idx and idx < end_idx) or use_full_pattern then
            if self.params.chars then
                for c in token:gmatch"." do table.insert(tokens, c) end
                table.insert(tokens, ' ')
            else
                table.insert(tokens, token)
            end
        end
        idx = idx + 1
    end

    if not use_full_pattern then
        table.insert(token_ids, vocab_map[first_arg] or self.params.unkIdx)
        if self.params.chars then table.insert(token_ids, vocab_map[' '] or self.params.unkIdx) end
    end

    for i = 1, #tokens do
        local token
        if not self.params.logRelations or #tokens <= 4 or i <= 2 or i > #tokens -2 then
            token = tokens[i]
        elseif i == 3 then
            token = '[' .. math.floor(torch.log(#tokens - 4)/torch.log(2)) .. ']'
        end
        if token then
            if self.params.doubleVocab then token = token .. '_' .. (arg1_first and '$ARG1' or '$ARG2') end
            local id = vocab_map[token] or self.params.unkIdx
            table.insert(token_ids, id)
            if id == self.params.unkIdx then self.out_vocab = self.out_vocab + 1 else self.in_vocab = self.in_vocab + 1 end
        end
    end

    if not use_full_pattern then table.insert(token_ids, vocab_map[second_arg] or self.params.unkIdx) end
    local pattern_tensor = torch.Tensor(token_ids)
    return pattern_tensor, #tokens
end

--- convert sentence to tac tensor using whole relation tensor ---
function SentenceClassifier:rel_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx, use_full_pattern)
    local rel_string
    if not use_full_pattern then
        local idx = 0
        local tokens = {}
        for token in string.gmatch(pattern_rel, "[^" .. self.params.delim .. "]+") do
            if self.params.tokenAppend ~= '' then token = token .. self.params.tokenAppend end
            if idx >= start_idx and idx < end_idx then table.insert(tokens, token) end
            idx = idx + 1
        end

        local first_arg = arg1_first and '$ARG1' or '$ARG2'
        if self.params.tokenAppend ~= '' then first_arg = first_arg .. self.params.tokenAppend end
        rel_string = first_arg .. ' '
        for i = 1, #tokens do
            if not self.params.logRelations or #tokens <= 4 or i <= 2 or i > #tokens - 2 then
                rel_string = rel_string .. tokens[i] .. ' '
            elseif i == 3 then
                rel_string = rel_string .. '[' .. math.floor(torch.log(#tokens - 4) / torch.log(2)) .. ']' .. ' '
            end
        end
        local second_arg = arg1_first and '$ARG2' or '$ARG1'
        if self.params.tokenAppend ~= '' then second_arg = second_arg .. self.params.tokenAppend end
        rel_string = rel_string .. second_arg
    else
        rel_string = pattern_rel
    end
    if self.params.appendEs then rel_string = rel_string .. "@es" end
    local id = -1
    local len = 0
    if vocab_map[rel_string] then
        id = vocab_map[rel_string]
        len = 1
        self.in_vocab = self.in_vocab + 1
    else
        self.out_vocab = self.out_vocab + 1
    end
    local pattern_tensor = torch.Tensor({id})
    return pattern_tensor, len
end

--- process a single line from a candidate file ---
function SentenceClassifier:process_line(line, vocab_map, dictionary)
    local query_id, tac_rel, sf_2, doc_info, start_1, end_1, start_2, end_2, pattern
    = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")

    if self.params.normalizeDigits and not self.params.fullPath then pattern = pattern:gsub("%d", "") end

    local tac_tensor = torch.Tensor({vocab_map[tac_rel] or self.params.unkIdx})

    -- we only want tokens between the two args
    local start_idx = tonumber(end_1)
    local end_idx = tonumber(start_2)
    local arg1_first = true
    if (start_idx > end_idx) then
        start_idx, end_idx, arg1_first = tonumber(end_2), tonumber(start_1), false
    end

    local pattern_tensor, seq_len
    if self.params.relations then
        pattern_tensor, seq_len = self:rel_tensor(arg1_first, pattern, vocab_map, start_idx, end_idx, self.params.fullPath)
    else
        pattern_tensor, seq_len = self:token_tensor(arg1_first, pattern, vocab_map, dictionary, start_idx, end_idx, self.params.fullPath)
    end

    pattern_tensor = pattern_tensor:view(1, pattern_tensor:size(1))
    tac_tensor = tac_tensor:view(1, tac_tensor:size(1))
    local enitity_pair = query_id .. '\t' .. sf_2
    local out_line = query_id .. '\t' .. tac_rel .. '\t' .. sf_2 .. '\t' .. doc_info .. '\t'
            .. start_1 .. '\t' .. end_1 .. '\t' .. start_2 .. '\t' .. end_2 .. '\t'

    return enitity_pair, pattern, out_line, pattern_tensor, tac_tensor, seq_len
end

--- process the candidate file and convert to torch ---
function SentenceClassifier:process_file(vocab_map, dictionary)
    local line_num = 0
    local max_seq = 0
    local data = {}
    print('Processing data')
    for line in io.lines(self.params.candidates) do
        local _, _, out_line, pattern_tensor, tac_tensor, seq_len = self:process_line(line, vocab_map, dictionary)
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
function SentenceClassifier:score_tac_relation(pattern_tensor, tac_tensor)
    if #self.text_encoder:findModules('nn.EncoderPool') > 0 then pattern_tensor = pattern_tensor:view(pattern_tensor:size(1), 1, pattern_tensor:size(2)) end
    if #self.kb_encoder:findModules('nn.EncoderPool') > 0 then tac_tensor = tac_tensor:view(tac_tensor:size(1), 1, tac_tensor:size(2)) end

    local tac_encoded = self.kb_encoder:forward(self:to_cuda(tac_tensor)):clone()
    local pattern_encoded = self.text_encoder:forward(self:to_cuda(pattern_tensor)):clone()

    if tac_encoded:dim() == 3 then tac_encoded = tac_encoded:view(tac_encoded:size(1), tac_encoded:size(3)) end
    if pattern_encoded:dim() == 3 then pattern_encoded = pattern_encoded:view(pattern_encoded:size(1), pattern_encoded:size(3)) end
    local x = { tac_encoded, pattern_encoded }

    local score = self:to_cuda(nn.CosineDistance())(x):double()
    --    local score = to_cuda(nn.Sum(2))(to_cuda(nn.CMulTable())(x)):double()
    return score
end

--- score the data returned by process_file ---
function SentenceClassifier:score_data(data, max_seq)
    print('Scoring data')
    -- open output file to write scored candidates file
    local out_file = io.open(self.params.outFile, "w")
    for seq_len = 1, math.min(max_seq, self.params.maxSeq) do
        if data[seq_len] then
            io.write('\rseq length : ' .. seq_len); io.flush()
            local seq_len_data = data[seq_len]
            --- batch
            --            local start = 1
            --            while start <= #seq_len_data do
            local pattern_tensor = nn.JoinTable(1)(seq_len_data.pattern_tensor)
            local tac_tensor = nn.JoinTable(1)(seq_len_data.tac_tensor)
            local scores = self:score_tac_relation(pattern_tensor, tac_tensor)
            local out_lines = seq_len_data.out_line
            for i = 1, #out_lines do
                local score = math.max(self.params.threshold, scores[i])
                out_file:write(out_lines[i] .. score .. '\n')
            end
        end
    end
    out_file:close()
end

function SentenceClassifier:load_maps()
    local vocab_map = {}
    for line in io.lines(self.params.vocabFile) do
        local token, id = string.match(line, "([^\t]+)\t([^\t]+)")
        if token and id then
            id = tonumber(id)
            if id > 1 then vocab_map[token] = id end
        end
    end
    local dictionary = {}
    if self.params.dictionary ~= '' then
        for line in io.lines(self.params.dictionary) do
            -- space seperated
            local en, es = string.match(line, "([^\t]+) ([^\t]+)")
            dictionary[es] = en
        end
    end
    return vocab_map, dictionary
end
