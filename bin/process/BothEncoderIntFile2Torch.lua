--
-- User: pat
-- Date: 8/10/15
--


require 'torch'

-- converts a 4 col mtx file to torch tensors
-- note : in data is assumed to be 0 indexed but all output is 1 indxed
local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-outFile', '', 'out file')
cmd:option('-delim', ' ', 'delimiter to split lines on')
cmd:option('-maxSeq', 999999, 'throw away sequences longer than this')
cmd:option('-minCount', 1, 'throw away tokens seen less than this many times')
cmd:option('-padIdx', 2, 'pad token')


local params = cmd:parse(arg)

print(params)

local data = {}
local max_row = 0
local max_row_token = 0
local max_col = 0
local max_col_token = 0
local num_lines = 0
local max_seq_len = 0
local length_counts = {}

print('Gathering token and sequence length stats')
-- get counts of all the tokens for filtering in frequent ones as well as counts of each seq length
for line in io.lines(params.inFile) do
    num_lines = num_lines + 1
    local row, row_tokens, col, col_tokens, label = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    local row_seq_len = 0
    for token in string.gmatch(row_tokens, "[^" .. params.delim .. "]+") do
        row_seq_len = row_seq_len + 1
        token = tonumber(token)
        max_row_token = math.max(token, max_row_token)
    end

    local col_seq_len = 0
    for token in string.gmatch(col_tokens, "[^" .. params.delim .. "]+") do
        col_seq_len = col_seq_len + 1
        token = tonumber(token)
        max_col_token = math.max(token, max_col_token)
    end

    local seq_len = math.max(row_seq_len, col_seq_len)
    if seq_len <= params.maxSeq then
        max_seq_len = math.max(seq_len, max_seq_len)
        length_counts[seq_len] = 1 + (length_counts[seq_len] or 0)
    end
end

print('Initializing tensors')
-- initialize all the tensors and seperate by seq length
for i = 1, max_seq_len do
    local count = length_counts[i]
    if count then
        local row_str_tensor = torch.Tensor(count, 1)
        local col_str_tensor = torch.Tensor(count, 1)
        local row_seq_tensor = torch.Tensor(count, i)
        local col_seq_tensor = torch.Tensor(count, i)
        local labelTensor = torch.Tensor(count, 1)
        data[i] = {row = row_str_tensor, col = col_str_tensor, row_seq = row_seq_tensor, col_seq = col_seq_tensor,
            label = labelTensor, count = 0, max_length = params.maxSeq, min_count = params.minCount}
    end
end

print ('Filling in tensors\n')
local lin_num = 0
for line in io.lines(params.inFile) do
    if (lin_num % 10000 == 0) then io.write('\rProcessing line number : '..lin_num); io.flush() end
    lin_num = lin_num + 1
    local row, row_tokens, col, col_tokens, label = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    local row_seq_len = 0
    local col_seq_len = 0
    local row_token_table = {}
    local col_token_table = {}
    for token in string.gmatch(row_tokens, "[^" .. params.delim .. "]+") do
        row_seq_len = row_seq_len + 1
        table.insert(row_token_table, token)
    end
    for token in string.gmatch(col_tokens, "[^" .. params.delim .. "]+") do
        col_seq_len = col_seq_len + 1
        table.insert(col_token_table, token)
    end

    local seq_len = math.max(row_seq_len, col_seq_len)
    -- pad the smaller sequence
    for i = #row_token_table, seq_len-1 do table.insert(row_token_table, params.padIdx) end
    for i = #col_token_table, seq_len-1 do table.insert(col_token_table, params.padIdx) end

    local len_data = data[seq_len]
    if len_data then
        local i = len_data.count + 1
        len_data.count = i
        for j, token in ipairs(row_token_table) do len_data.row_seq[i][j] = token end
        for j, token in ipairs(col_token_table) do len_data.col_seq[i][j] = token end
        len_data.row[i] = row
        len_data.col[i] = col
        len_data.label[i] = label

        max_row = math.max(tonumber(row), max_row)
        max_col = math.max(tonumber(col), max_col)
    end
end

print('\nSaving data')
-- attach meta data
data.num_rows = max_row
data.num_cols = max_col
data.num_row_tokens = max_row_token
data.num_col_tokens = max_col_token
data.max_length = math.min(max_seq_len, params.maxSeq)
data.min_count = params.minCount


torch.save(params.outFile, data)
print(string.format('num rows = %d\t num row tokens = %d', max_row, max_row_token))
print(string.format('num cols = %d\t num col tokens = %d', max_col, max_col_token))
