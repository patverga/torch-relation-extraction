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

--[[
    Takes a tac candidate file, tab seperated vocab idx file, and a trained uschema encoder model
    and exports a scored candidtate file to outfile
]]--
local cmd = torch.CmdLine()
cmd:option('-positiveTrainCandidates', '', 'input candidate file')
cmd:option('-negativeTrainCandidates', '', 'input candidate file')
cmd:option('-testCandidates', '', 'input candidate file')
cmd:option('-outFile', '', 'scored candidate out file')
cmd:option('-vocabFile', ' ', 'txt file containing vocab-index map')
cmd:option('-maxSeq', 999999, 'throw away sequences longer than this')
cmd:option('-model', '', 'a trained model that will be used to score candidates')
cmd:option('-delim', ' ', 'delimiter to split lines on')
cmd:option('-threshold', 0, 'scores below this threshold will be set to -1e100')
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')
cmd:option('-relations', false, 'Use full relation vectors instead of tokens')
cmd:option('-logRelations', false, 'Use log relation vectors instead of tokens')
cmd:option('-doubleVocab', false, 'double vocab so that tokens to the right of ARG1 are different then to the right of ARG2')
cmd:option('-appendEs', false, 'append @es to end of relation')
cmd:option('-normalizeDigits', false, 'map all digits to #')
cmd:option('-tokenAppend', '', 'append this to the end of each token')
cmd:option('-use_cosine', false, 'whether to use cosineembedding loss when post-training the scaling of the relation vectors')


local params = cmd:parse(arg)
local function to_cuda(x) return params.gpuid >= 0 and x:cuda() or x end
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local in_vocab = 0
local out_vocab = 0

--- convert sentence to tac tensor using tokens ---
local function token_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx)
    local idx = 0
    local i = 0
    local token_ids = {}
    local tokens = {}
    for token in string.gmatch(pattern_rel, "[^" .. params.delim .. "]+") do
        if params.tokenAppend ~= '' then token = token .. params.tokenAppend end
        if idx >= start_idx and idx < end_idx then table.insert(tokens, token) end
        idx = idx + 1
    end
    local first_arg = arg1_first and '$ARG1' or '$ARG2'
    if params.tokenAppend ~= '' then first_arg = first_arg .. params.tokenAppend end
    table.insert(token_ids, vocab_map[first_arg] or 1)
    for i = 1, #tokens do
        local token
        if not params.logRelations or #tokens <= 4 or i <= 2 or i > #tokens -2 then
            token = tokens[i]
        elseif i == 3 then
            token = '[' .. math.floor(torch.log(#tokens - 4)/torch.log(2)) .. ']'
        end
        if token then
            if params.doubleVocab then token = token .. '_' .. (arg1_first and '$ARG1' or '$ARG2') end
            local id = vocab_map[token] or 1
            table.insert(token_ids, id)
            if id == 1 then out_vocab = out_vocab + 1 else in_vocab = in_vocab + 1 end
        end
    end
    local second_arg = arg1_first and '$ARG2' or '$ARG1'
    if params.tokenAppend ~= '' then second_arg = second_arg .. params.tokenAppend end
    table.insert(token_ids, vocab_map[second_arg] or 1)
    local pattern_tensor = torch.Tensor(token_ids)
    return pattern_tensor, #tokens
end

--- convert sentence to tac tensor using whole relation tensor ---
local function rel_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx)
    local idx = 0
    local tokens = {}
    for token in string.gmatch(pattern_rel, "[^" .. params.delim .. "]+") do
        if params.tokenAppend ~= '' then token = token .. params.tokenAppend end
        if idx >= start_idx and idx < end_idx then table.insert(tokens, token) end
        idx = idx + 1
    end

    local first_arg = arg1_first and '$ARG1' or '$ARG2'
    if params.tokenAppend ~= '' then first_arg = first_arg .. params.tokenAppend end
    local rel_string = first_arg .. ' '
    for i = 1, #tokens do
        if not params.logRelations or #tokens <= 4 or i <= 2 or i > #tokens -2 then
            rel_string = rel_string .. tokens[i] .. ' '
        elseif i == 3 then
            rel_string = rel_string .. '[' .. math.floor(torch.log(#tokens - 4)/torch.log(2)) .. ']' .. ' '
        end
    end
    local second_arg = arg1_first and '$ARG2' or '$ARG1'
    if params.tokenAppend ~= '' then second_arg = second_arg .. params.tokenAppend end
    rel_string = rel_string .. second_arg
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
local function process_line(line, vocab_map)
    local query_id, tac_rel, sf_2, doc_info, start_1, end_1, start_2, end_2, pattern_rel
    = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")

    if params.normalizeDigits then pattern_rel = pattern_rel:gsub("%^a", "") end

    local tac_tensor = torch.Tensor({vocab_map[tac_rel] or 1})

    -- we only want tokens between the two args
    local start_idx = tonumber(end_1)
    local end_idx = tonumber(start_2)
    local arg1_first = true
    if (start_idx > end_idx) then
        start_idx, end_idx, arg1_first = tonumber(end_2), tonumber(start_1), false
    end

    local pattern_tensor, seq_len
    if params.relations then
        pattern_tensor, seq_len = rel_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx)
    else
        pattern_tensor, seq_len = token_tensor(arg1_first, pattern_rel, vocab_map, start_idx, end_idx)
    end

    pattern_tensor = pattern_tensor:view(1, pattern_tensor:size(1))
    tac_tensor = tac_tensor:view(1, tac_tensor:size(1))
    local out_line = query_id .. '\t' .. tac_rel .. '\t' .. sf_2 .. '\t' .. doc_info .. '\t'
            .. start_1 .. '\t' .. end_1 .. '\t' .. start_2 .. '\t' .. end_2 .. '\t'

    return out_line, pattern_tensor, tac_tensor, seq_len
end

--- process the candidate file and convert to torch ---
local function process_file(candidates, vocab_map)
    local line_num = 0
    local max_seq = 0
    local data = {}
    print('Processing data')
    for line in io.lines(candidates) do
        local out_line, pattern_tensor, tac_tensor, seq_len = process_line(line, vocab_map)
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
local function score_tac_relation(text_encoder, kb_col_table, pattern_tensor, tac_tensor, transform_net)
    local tac_encoded = kb_col_table:forward(to_cuda(tac_tensor)):clone()
    local pattern_encoded = text_encoder:forward(to_cuda(pattern_tensor)):clone()

    if tac_encoded:dim() == 3 then tac_encoded = tac_encoded:view(tac_encoded:size(1), tac_encoded:size(3)) end
    if pattern_encoded:dim() == 3 then pattern_encoded = pattern_encoded:view(pattern_encoded:size(1), pattern_encoded:size(3)) end
    local score = transform_net({ tac_encoded, pattern_encoded }):double()
    return score
end

--- score the data returned by process_file ---
local function score_data(data, max_seq, text_encoder, kb_col_table, transform_net)
    print('Scoring data')
    -- open output file to write scored candidates file
    local out_file = io.open(params.outFile, "w")
    local score_transformer = params.use_cosine and to_cuda(nn.CosineDistance()) or to_cuda(nn.Sequential():add(nn.CMulTable()):add(nn.Sum(2)))
    for seq_len = 1, math.min(max_seq, params.maxSeq) do
        if data[seq_len] then
            io.write('\rseq length : ' .. seq_len); io.flush()
            local seq_len_data = data[seq_len]
            --- batch
--            local start = 1
--            while start <= #seq_len_data do
            local pattern_tensor = nn.JoinTable(1)(seq_len_data.pattern_tensor)
            local tac_tensor = nn.JoinTable(1)(seq_len_data.tac_tensor)
            local scores = score_tac_relation(text_encoder, kb_col_table, pattern_tensor, tac_tensor, transform_net)
            local out_lines = seq_len_data.out_line
            for i = 1, #out_lines do
                local score = scores[i] > params.threshold and scores[i] or 0
                out_file:write(out_lines[i] .. score .. '\n')
            end
        end
    end
    out_file:close()
end



local function gen_batches(tac_tensors, pattern_tensors, label_tensors, batch_size)
    local batches = {}
    local start = 1
    local rand_order = torch.randperm(tac_tensors:size(1)):long()
    while start <= tac_tensors:size(1) do
        local size = math.min(batch_size, tac_tensors:size(1) - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local tac_tensor_batch = tac_tensors:index(1, batch_indices)
        local pattern_tensor_batch = pattern_tensors:index(1, batch_indices)
        local label_batch = label_tensors:index(1, batch_indices)
        local batch = { tac_tensor_batch, pattern_tensor_batch }
        table.insert(batches, { data = batch, label = label_batch })
        start = start + size
    end
    return batches
end

local function train_transform(kb_rel_encoder, text_encoder, vocab_map)

    local max_epochs = 25
    local margin = .5
    local batch_size = 512
    local learning_rate = 0.1
    local stop_threshold = .0001
    local clamp_max = 1000
    local clamp_min = 0.001

    local pos_data, pos_max_seq = process_file(params.positiveTrainCandidates, vocab_map)
    local neg_data, neg_max_seq = process_file(params.negativeTrainCandidates, vocab_map)

    local tac_tensors = {}
    local pattern_tensors = {}
    local label_tensors = {}

    local count = 0
    for seq_len = 1, math.min(pos_max_seq, params.maxSeq) do
        if pos_data[seq_len] then
            local seq_len_data = pos_data[seq_len]
            local pattern_tensor = nn.JoinTable(1)(seq_len_data.pattern_tensor)
            local tac_tensor = nn.JoinTable(1)(seq_len_data.tac_tensor)
            local tac_encoded = kb_rel_encoder:forward(to_cuda(tac_tensor)):clone()
            table.insert(tac_tensors, tac_encoded)
            local pattern_encoded = text_encoder:forward(to_cuda(pattern_tensor)):clone()
            table.insert(pattern_tensors, pattern_encoded)
            table.insert(label_tensors, to_cuda(torch.Tensor(pattern_encoded:size(1)):fill(1)))
            count = count + pattern_encoded:size(1)
        end
    end
    for seq_len = 1, math.min(neg_max_seq, params.maxSeq) do
        if neg_data[seq_len] then
            local seq_len_data = neg_data[seq_len]
            local pattern_tensor = nn.JoinTable(1)(seq_len_data.pattern_tensor)
            local tac_tensor = nn.JoinTable(1)(seq_len_data.tac_tensor)
            local tac_encoded = kb_rel_encoder:forward(to_cuda(tac_tensor)):clone()
            table.insert(tac_tensors, tac_encoded)
            local pattern_encoded = text_encoder:forward(to_cuda(pattern_tensor)):clone()
            table.insert(pattern_tensors, pattern_encoded)
            count = count + pattern_encoded:size(1)
            local value_to_fill = use_cosine and -1 or 0
            table.insert(label_tensors, to_cuda(torch.Tensor(pattern_encoded:size(1)):fill(value_to_fill)))
        end
    end

    print('Found ' .. count .. ' training examples.')
    local join_tabel = nn.JoinTable(1)
    to_cuda(join_tabel)
    tac_tensors = join_tabel(tac_tensors):clone()
    pattern_tensors = join_tabel(pattern_tensors):clone()
    label_tensors = join_tabel(label_tensors):clone()


    local criterion = use_cosine and  nn.CosineEmbeddingCriterion(margin) or nn.BCECriterion()
    local cmulTac = nn.CMul(pattern_tensors:size(2))
    local transform_net = nn:Sequential():add(nn.ParallelTable():add(cmulTac):add(nn.Identity()))
    if(not use_cosine) then
        local norm = nn.ParallelTable():add(nn.Normalize(2)):add(nn.Normalize(2))
        local dot = nn.Sequential():add(norm):add(nn.CMulTable()):add(nn.Sum(2)):add(nn.Mul())
        transform_net:add(dot)
        transform_net:add(nn.Sigmoid())
    end
    print(transform_net)
    to_cuda(transform_net)
    to_cuda(criterion)
    local parameters, gradParameters = transform_net:getParameters()

    -- loop over data batches
    local last_epoch_error = 10000
    local total_error = 1000
    local epoch = 1
    while ((math.abs(last_epoch_error - total_error) / last_epoch_error) > stop_threshold and epoch < max_epochs) do
        print ('Starting training epoch ' .. epoch)
        last_epoch_error = total_error
        total_error = 0
        local startTime = sys.clock()
        local batches = gen_batches(tac_tensors, pattern_tensors, label_tensors, batch_size)
        for i = 1, #batches do
            transform_net:zeroGradParameters()
            local x = batches[i].data
            local y = batches[i].label
            local pred = transform_net:forward(x)
            total_error = total_error + criterion:forward(pred, y)
            local df_do = criterion:backward(pred, y)
            transform_net:backward(x, df_do)
            transform_net:updateParameters(learning_rate / math.sqrt(epoch))
            -- force parameters to be > 0
            parameters:clamp(clamp_min, clamp_max)
--            transform_net:accUpdateGradParameters(x, df_do, learning_rate)
            if (i % 10 == 0) then
                io.write(string.format('\r%.2f percent complete\tspeed = %.2f examples/sec\terror = %.4f',
                    100 * i / (#batches), (i * batch_size) / (sys.clock() - startTime), (total_error / i)))
                io.flush()
            end
        end
        print ('\nFinished epoch ' .. epoch .. '. Total error : '.. total_error)
        epoch = epoch + 1
    end

    if(use_cosine) then
        transform_net:add(nn.CosineDistance()) 
    end
    return transform_net
end


---- main

print('Deserializing model')
local model = torch.load(params.model)
local kb_rel_encoder = to_cuda(model.kb_col_table ~= nil and model.kb_col_table or model.encoder):clone()
local text_encoder = to_cuda(model.text_encoder ~= nil and model.text_encoder or model.encoder):clone()
kb_rel_encoder:evaluate()
text_encoder:evaluate()

-- load the vocab map to memory
local vocab_map = {}
for line in io.lines(params.vocabFile) do
    local token, id = string.match(line, "([^\t]+)\t([^\t]+)")
    if token then vocab_map[token] = tonumber(id) end
end

local transform_net = train_transform(kb_rel_encoder, text_encoder, vocab_map)
local data, max_seq = process_file(params.testCandidates, vocab_map)
score_data(data, max_seq, text_encoder, kb_rel_encoder, transform_net)

print ('\nDone, found ' .. in_vocab .. ' in vocab tokens and ' .. out_vocab .. ' out of vocab tokens.')
