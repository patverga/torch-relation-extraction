--
-- User: pat
-- Date: 8/10/15
--


require 'torch'
require 'nn'

-- note : in data is assumed to be 0 indexed but all output is 1 indxed
local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-outFile', '', 'out file')
cmd:option('-delim', ' ', 'delimiter to split lines on')
cmd:option('-maxSeq', 50, 'throw away sequences longer than this')
cmd:option('-maxCount', 10000, 'throw away eps with more than this many relations')
cmd:option('-padToken', 1, 'Pad all sequences with this token up to maxSeq')


local params = cmd:parse(arg)

print(params)

local max_ep = 0
local max_token = 0
local num_rows = 0
local ep_rels = {}

print('Seperating relations by ep')
for line in io.lines(params.inFile) do
    num_rows = num_rows + 1
    local e1, e2, ep, rel, token_str, label = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    local tokens = {}
    for token in string.gmatch(token_str, "[^" .. params.delim .. "]+") do
        -- TODO this is adding 1 to deal with pad token
        token = tonumber(token) + 1
        table.insert(tokens, token)
        local ep_num = tonumber(ep)
        max_token = math.max(token, max_token)
        max_ep = math.max(ep_num, max_ep)
    end
    if #tokens <= params.maxSeq then
        ep_rels[ep] = ep_rels[ep] or {}
        for i = #tokens, params.maxSeq-1 do table.insert(tokens, params.padToken) end
        table.insert(ep_rels[ep], torch.Tensor(tokens):view(1, #tokens))
    end
    if (num_rows % 10000 == 0) then io.write('\rProcessing line number : '..num_rows); io.flush() end
end

print('\nJoining tensors')
local join = nn.JoinTable(1)
local rel_counts = {}
local ep_counts = {}
local max_count = 0
for ep, rel_table in pairs(ep_rels) do
    max_count = math.max(max_count, #rel_table)
    local rel_tensor = join(rel_table):clone()
    rel_counts[#rel_table] = rel_counts[#rel_table] or {}
    ep_counts[#rel_table] = ep_counts[#rel_table] or {}
    table.insert(rel_counts[#rel_table], rel_tensor:view(1, rel_tensor:size(1), rel_tensor:size(2)))
    table.insert(ep_counts[#rel_table], ep)
end

ep_rels = nil
local data = {}
for i = 1, math.min(params.maxCount, max_count) do
    if rel_counts[i] then
        local epTensor = torch.Tensor(ep_counts[i])
        local seqTensor = join(rel_counts[i]):clone()
        data[i] = { ep = epTensor, seq = seqTensor, count = 0, num_eps = max_ep, num_tokens = max_token }
    end
end

print('\nSaving data')
-- attach meta data
data.num_eps = max_ep
data.num_tokens = max_token
data.max_length = params.max_count


torch.save(params.outFile, data)
print(string.format('num rows = %d\t num unique tokens = %d', num_rows, max_token))
