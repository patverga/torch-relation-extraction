--
-- User: pat
-- Date: 8/10/15
--


require 'torch'


-- note : in data is assumed to be 0 indexed but all output is 1 indxed
local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-outFile', '', 'out file')
cmd:option('-delim', ' ', 'delimiter to split lines on')
cmd:option('-maxSeq', 999999, 'throw away sequences longer than this')
cmd:option('-minCount', 1, 'throw away tokens seen less than this many times')


local params = cmd:parse(arg)

print(params)

local data = {}
local max_ent = 0
local max_ep = 0
local max_rel = 0
local max_token = 0
local num_rows = 0
local max_len = 0
local length_counts = {}

print('Gathering token and sequence length stats')
-- get counts of all the tokens for filtering in frequent ones as well as counts of each seq length
for line in io.lines(params.inFile) do
    num_rows = num_rows + 1
    local e1, e2, ep, rel, tokens, label = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    local seq_len = 0
    for token in string.gmatch(tokens, "[^" .. params.delim .. "]+") do
        seq_len = seq_len + 1
        token = tonumber(token)
--        print(token)
        if token > max_token then max_token = token end
    end
    if seq_len > max_len then max_len = seq_len end
    if seq_len <= params.maxSeq then
        length_counts[seq_len] = 1 + (length_counts[seq_len] or 0)
    end
end

print('Initializing tensors')
-- initialize all the tensors and seperate by seq length
for i = 1, max_len do
    local count = length_counts[i]
    if count then
        local e1Tensor = torch.Tensor(count, 1)
        local e2Tensor = torch.Tensor(count, 1)
        local epTensor = torch.Tensor(count, 1)
        local relTensor = torch.Tensor(count, 1)
        local seqTensor = torch.Tensor(count, i)
        local labelTensor = torch.Tensor(count, 1)
        data[i] = {ep = epTensor, e1 = e1Tensor, e2 = e2Tensor, rel = relTensor, seq = seqTensor, label = labelTensor, count = 0 }
    end
end

print ('Filling in tensors\n')
local lin_num = 0
for line in io.lines(params.inFile) do
    if (lin_num % 100000 == 0) then io.write('\r'..lin_num); io.flush() end
    lin_num = lin_num + 1
    local e1, e2, ep, rel, tokens, label = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    local seq_len = 0
    local token_table = {}
    for token in string.gmatch(tokens, "[^" .. params.delim .. "]+") do
        seq_len = seq_len + 1
        table.insert(token_table, token)
    end

    local len_data = data[seq_len]
    if len_data then
        local i = len_data.count + 1
        len_data.count = i
        for j, token in ipairs(token_table) do len_data.seq[i][j] = token end

        len_data.e1[i] = e1
        len_data.e2[i] = e2
        len_data.ep[i] = ep
        len_data.rel[i] = rel
        len_data.label[i] = label
        e1 = tonumber(e1)
        e2 = tonumber(e2)
        ep = tonumber(ep)
        rel = tonumber(rel)
        if e1 > max_ent then max_ent = e1 end
        if e2 > max_ent then max_ent = e2 end
        if ep > max_ep then max_ep = ep end
        if rel > max_rel then max_rel = rel end
    end
end

print('\nSaving data')
-- attach meta data
data.num_rels = max_rel
data.num_eps = max_ep
data.num_ents = max_ent
data.num_tokens = max_token
data.max_length = params.maxSeq
data.min_count = params.minCount
for i = 1, #data do
    if data[i] then
        data[i].num_rels = max_rel
        data[i].num_eps = max_ep
        data[i].num_ents = max_ent
        data[i].num_tokens = max_token
        data[i].max_length = params.maxSeq
        data[i].min_count = params.minCount
    end
end

torch.save(params.outFile, data)
print(string.format('num rows = %d\t num unique tokens = %d', num_rows, max_token))
