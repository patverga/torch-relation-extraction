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
cmd:option('-minCount', 1, 'throw away tokens seen less than this many times')
cmd:option('-padIdx', 2, 'pad token')


local params = cmd:parse(arg)

print(params)

local max_row = 0
local max_token = 0
local num_rows = 0
local cols_for_row = {}

print('Seperating columns by rows')
for line in io.lines(params.inFile) do
    num_rows = num_rows + 1
    local row, row_tokens, col, col_tokens, label = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    local tokens = {}
    for token in string.gmatch(col_tokens, "[^" .. params.delim .. "]+") do
        token = tonumber(token)
        table.insert(tokens, token)
        local row_num = tonumber(row)
        max_token = math.max(token, max_token)
        max_row = math.max(row_num, max_row)
    end
    if #tokens <= params.maxSeq then
        cols_for_row[row] = cols_for_row[row] or {}
        for i = #tokens, params.maxSeq-1 do table.insert(tokens, params.padIdx) end
        table.insert(cols_for_row[row], torch.Tensor(tokens):view(1, #tokens))
    end
    if (num_rows % 10000 == 0) then io.write('\rProcessing line number : '..num_rows); io.flush() end
end

print('\nJoining tensors')
local join = nn.JoinTable(1)
local col_counts = {}
local row_counts = {}
local max_count = 0
local row_num = 0
for row, col_table in pairs(cols_for_row) do
    max_count = math.max(max_count, #col_table)
    local rel_tensor = join(col_table):clone()
    col_counts[#col_table] = col_counts[#col_table] or {}
    row_counts[#col_table] = row_counts[#col_table] or {}
    table.insert(col_counts[#col_table], rel_tensor:view(1, rel_tensor:size(1), rel_tensor:size(2)))
    table.insert(row_counts[#col_table], row)
    row_num = row_num +1
    if (row_num % 100 == 0) then io.write('\rProcessing row number : '.. row_num); io.flush() end
end
cols_for_row = nil

local data = { num_rows = max_row, num_col_tokens = max_token, max_length = params.max_count }
for i = 1, math.min(params.maxCount, max_count) do
    if col_counts[i] then
        local row_tensor = torch.Tensor(row_counts[i])
        local col_seq_tensor = join(col_counts[i]):clone()
        data[i] = {row = row_tensor, col_seq = col_seq_tensor, count = 0, max_length = params.maxSeq }
    end
end

row_counts = nil; col_counts = nil;

print('\nSaving data')
torch.save(params.outFile, data)
print(string.format('num rows = %d\t num unique tokens = %d', num_rows, max_token))