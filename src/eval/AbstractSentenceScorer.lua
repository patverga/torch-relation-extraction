--
-- User: pat
-- Date: 2/5/16
--

package.path = package.path .. ";src/?.lua;src/nn-modules/?.lua;src/eval/?.lua"


local AbstractSentenceScorer = torch.class('AbstractSentenceScorer')

function AbstractSentenceScorer:__init(params, net, kb_encoder, text_encoder)
    self.params = params
    self.net = self:to_cuda(net)
    self.kb_encoder = self:to_cuda(kb_encoder)
    self.text_encoder = self:to_cuda(text_encoder)
    self.in_vocab = 0
    self.out_vocab = 0
end

function AbstractSentenceScorer:to_cuda(x) return self.params.gpuid >= 0 and x:cuda() or x end

function AbstractSentenceScorer:run()
    -- process the candidate file
    local data = self:process_file(self:load_maps())
    local max_scores, max_score, min_score, out_lines = self:score_data(data)
    self:write_output(max_scores, max_score, min_score, out_lines)
    print ('\nDone, found ' .. self.in_vocab .. ' in vocab tokens and ' .. self.out_vocab .. ' out of vocab tokens.')
end

function AbstractSentenceScorer:load_maps()
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
    local tac_map = {}
    if self.params.tacLabelMap ~= '' then
        for line in io.lines(self.params.tacLabelMap) do
            local token, id = string.match(line, "([^\t]+)\t([^\t]+)")
            if token and id then
                id = tonumber(id)
                if id > 1 then tac_map[token] = id end
            end
        end
    else
        tac_map = vocab_map
    end
    return vocab_map, tac_map, dictionary
end

--- process the candidate file and convert to torch ---
function AbstractSentenceScorer:process_file(pattern_map, tac_map, dictionary)
    local line_num = 0
    local max_seq = 0
    local data = {}
    print('Processing data')
    for line in io.lines(self.params.candidates) do
        local ep, _, out_line, pattern_tensor, tac_tensor, seq_len = self:process_line(line, pattern_map, tac_map, dictionary)
        if seq_len <= self.params.maxSeq then
            max_seq = math.max(seq_len, max_seq)
            if not data[seq_len] then data[seq_len] = {out_line={}, pattern_tensor={}, tac_tensor={}, ep={}} end
            local seq_len_data = data[seq_len]
            table.insert(seq_len_data.out_line, out_line)
            table.insert(seq_len_data.pattern_tensor, pattern_tensor)
            table.insert(seq_len_data.tac_tensor, tac_tensor)
            table.insert(seq_len_data.ep, ep)
        end
        line_num = line_num + 1
        if line_num % 10000 == 0 then io.write('\rline : ' .. line_num); io.flush() end
    end
    print ('\rProcessed ' .. line_num .. ' lines')
    return data
end


--- process a single line from a candidate file ---
function AbstractSentenceScorer:process_line(line, pattern_map, tac_map, dictionary)
    local query_id, tac_rel, sf_2, doc_info, start_1, end_1, start_2, end_2, pattern
    = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")

    if self.params.normalizeDigits ~= '' and not self.params.fullPath then pattern = pattern:gsub("%d", self.params.normalizeDigits) end

    local tac_tensor = torch.Tensor({tac_map[tac_rel] or self.params.unkIdx})

    -- we only want tokens between the two args
    local start_idx = tonumber(end_1)
    local end_idx = tonumber(start_2)
    local arg1_first = true
    if (start_idx > end_idx) then
        start_idx, end_idx, arg1_first = tonumber(end_2), tonumber(start_1), false
    end

    local pattern_tensor, seq_len
    if self.params.relations then
        pattern_tensor, seq_len = self:rel_tensor(arg1_first, pattern, pattern_map, start_idx, end_idx, self.params.fullPath)
    else
        pattern_tensor, seq_len = self:token_tensor(arg1_first, pattern, pattern_map, dictionary, start_idx, end_idx, self.params.fullPath)
    end

    pattern_tensor = pattern_tensor:view(1, pattern_tensor:size(1))
    tac_tensor = tac_tensor:view(1, tac_tensor:size(1))
    local enitity_pair = query_id .. '\t' .. sf_2
    local out_line = query_id .. '\t' .. tac_rel .. '\t' .. sf_2 .. '\t' .. doc_info .. '\t'
            .. start_1 .. '\t' .. end_1 .. '\t' .. start_2 .. '\t' .. end_2 .. '\t'

    return enitity_pair, pattern, out_line, pattern_tensor, tac_tensor, seq_len
end



--- convert sentence to tac tensor using tokens ---
function AbstractSentenceScorer:token_tensor(arg1_first, pattern_rel, vocab_map, dictionary, start_idx, end_idx, use_full_pattern)
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
                if self.params.normalizeDigits ~= '' and self.params.fullPath then token = token:gsub("%d", self.params.normalizeDigits) end
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
    return pattern_tensor, #token_ids
end

--- convert sentence to tac tensor using whole relation tensor ---
function AbstractSentenceScorer:rel_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx, use_full_pattern)
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

--- score the data returned by process_file ---
function AbstractSentenceScorer:score_data(data)
    print('Scoring data')

    local ep_tac_scores = self.params.poolWeight > 0 and self:pooled_scores(data) or nil

    -- open output file to write scored candidates file
    local max_scores = {}
    local max_score = -1e8
    local min_score = 1e8
    local out_lines = {}
    for seq_len, sub_data in pairs(data) do
        if torch.type(sub_data) == 'table' then
            io.write('\rseq length : ' .. seq_len); io.flush()
            local pattern_tensor = nn.JoinTable(1)(sub_data.pattern_tensor)
            local tac_tensor = nn.JoinTable(1)(sub_data.tac_tensor)
            local scores = self:score_tac_relation(pattern_tensor, tac_tensor)
            for i = 1, scores:size(1) do
                local ep = sub_data.ep[i]
                local tac_idx = tac_tensor[i][1]
                local score = math.max(self.params.threshold, scores[i])
                if ep_tac_scores then score = ((1.0-self.params.poolWeight)*score) + (self.params.poolWeight * ep_tac_scores[ep][tac_idx]) end
                max_score = math.max(score, max_score)
                min_score = math.min(score, min_score)
                if not max_scores[ep] then max_scores[ep] = {}; out_lines[ep] = {} end
                if self.params.scoring == 'max' then
                    if not max_scores[ep][tac_idx] or score > max_scores[ep][tac_idx] then
                        max_scores[ep][tac_idx] = score
                        out_lines[ep][tac_idx] = sub_data.out_line[i]
                    end
                elseif self.params.scoring == 'mean' then
                    if not max_scores[ep][tac_idx] then max_scores[ep][tac_idx] = {}; out_lines[ep][tac_idx] = {}; end
                        table.insert(max_scores[ep][tac_idx], score)
                        table.insert(out_lines[ep][tac_idx], sub_data.out_line[i])
                else
                    max_scores[ep][#max_scores[ep]+1] = score
                    out_lines[ep][#out_lines[ep]+1] = sub_data.out_line[i]
                end
            end
        end
    end
    return max_scores, max_score, min_score, out_lines
end

--- pool all relations for each entity pair and score them together
function AbstractSentenceScorer:pooled_scores(data)
    -- seperate data by eps
    local ep_data = {}
    for _, sub_data in pairs(data) do
        if torch.type(sub_data) == 'table' then
            for i = 1, #sub_data.tac_tensor do
                local ep = sub_data.ep[i]
                local tac_idx = sub_data.tac_tensor[i][1][1]
                if ep_data[ep] == nil then ep_data[ep] = {} end
                if ep_data[ep][tac_idx] == nil then ep_data[ep][tac_idx] = {} end
                local padded_pattern = sub_data.pattern_tensor[i]
                if padded_pattern:size(2) < self.params.maxSeq then
                    local padding = padded_pattern:clone():resize(1,self.params.maxSeq-padded_pattern:size(2)):fill(self.params.padIdx)
                    padded_pattern = padded_pattern:cat(padding)
                end
                table.insert(ep_data[ep][tac_idx], padded_pattern:view(1,1,-1))
            end
        end
    end
    -- group data by number of relations for batching
    local pattern_count_data = {}
    for ep, tac_indices in pairs(ep_data) do
        for tac_idx, pattern_table in pairs(tac_indices) do
            local pattern_tensor = nn.JoinTable(2)(pattern_table)
            local count = pattern_tensor:size(2)
            if not pattern_count_data[count] then pattern_count_data[count] = {pattern_tensor={}, tac_tensor={}, ep={}} end
            table.insert(pattern_count_data[count].pattern_tensor, pattern_tensor)
            table.insert(pattern_count_data[count].tac_tensor, torch.Tensor(1,1):fill(tac_idx))
            table.insert(pattern_count_data[count].ep, ep)
        end
    end
    -- get a score for each ep,pattern
    local ep_tac_scores = {}
    for _, sub_data in pairs(pattern_count_data) do
        local pattern_tensor = nn.JoinTable(1)(sub_data.pattern_tensor)
        local tac_tensor = nn.JoinTable(1)(sub_data.tac_tensor)
        local scores = self:score_tac_relation(pattern_tensor, tac_tensor)
        for i = 1, scores:size(1) do
            if ep_tac_scores[sub_data.ep[i]] == nil then ep_tac_scores[sub_data.ep[i]] = {} end
            ep_tac_scores[sub_data.ep[i]][tac_tensor[i][1]] = scores[i]
        end
    end
    return ep_tac_scores
end

function AbstractSentenceScorer:score_tac_relation(pattern_tensor, tac_tensor)
    print ('This class is abstract - Must implement this method.')
    os.exit()
end

function AbstractSentenceScorer:write_output(max_scores, max_score, min_score, out_lines)
    local out_file = io.open(self.params.outFile, "w")
    local score_range = max_score - min_score
    for ep, tac_tensors in pairs(out_lines) do
        for tac_idx, out_line in pairs(tac_tensors) do
            if self.params.scoring == 'mean' then
                local score = torch.Tensor(max_scores[ep][tac_idx]):mean()
                for i = 1, #out_line do out_file:write(out_line[i] .. ((score-min_score)/score_range) .. '\n') end
            else
                local score = max_scores[ep][tac_idx]
                out_file:write(out_line .. ((score-min_score)/score_range) .. '\n')
            end
        end
    end
    out_file:close()
end

