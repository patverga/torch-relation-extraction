require 'nn'

--[[
    Given a file processed by IntFile2PoolRelationsTorch.lua mapping entity pairs to a set of relations
    creates a new file mapping a single relation to a set of relations
  ]]--

local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-outFile', '', 'out file')
cmd:option('-kbMap', '', 'out file')
cmd:option('-kbOnly', false, 'only use kb relations as the single relation')
cmd:option('-maxSamples', 0, 'maximum number of samples to take for a given entity pair - 0 = all')
local params = cmd:parse(arg)


local kb_rels = {}
if params.kbMap ~= '' then
    io.write('Loading kb map... ')
    for line in io.lines(params.kbMap) do
        local token, id = string.match(line, "([^\t]+)\t([^\t]+)")
        if token and id then
            id = tonumber(id)
            if id > 1 then table.insert(kb_rels, torch.Tensor(1,1):fill(id)) end
        end
    end
    print('Done')
end

-- DATA
print('loading file: ', params.inFile)
local original_data = torch.load(params.inFile)
-- prepare new data object dNew
local reformatted_data = {num_cols = original_data.num_rels, num_rows = original_data.num_eps,
    num_col_tokens = original_data.num_tokens, num_row_tokens = original_data.num_tokens}


local function extract_example(ep_data, i)
    local row = ep_data.rel:select(2, i)
    local row_seq = ep_data.seq:select(2, i)
    local range
    if i == 1 then
        range = torch.range(i+1, ep_data.rel:size(2))
    elseif i == ep_data.rel:size(2) then
        range = torch.range(1, ep_data.rel:size(2)-1)
    else
        range = torch.range(1, i-1):cat(torch.range(i+1, ep_data.rel:size(2)))
    end
    local col = ep_data.rel:index(2, range:long())
    col = col:view(col:size(1), col:size(2))
    local col_seq = ep_data.seq:index(2, range:long())

    return col, row, col_seq, row_seq
end

-- for each r in the relation set, create #r examples - take one out and leave the rest
local function extract_relations(ep_data)
    local col_table, row_table, col_seq_table, row_seq_table = {}, {}, {}, {}
    local shuffle = torch.randperm(ep_data.rel:size(2))
    if params.maxSamples > 0 then shuffle = shuffle:narrow(1, 1, math.min(shuffle:size(1), params.maxSamples)) end
    for i = 1, shuffle:size(1) do
        local col, row, col_seq, row_seq = extract_example(ep_data, shuffle[i])
        table.insert(col_table, col);  table.insert(col_seq_table, col_seq)
        table.insert(row_table, row);  table.insert(row_seq_table, row_seq)
    end
    return col_table, row_table, col_seq_table, row_seq_table
end

-- for each kb relation in the relation set, create #kb examples - take one out and leave the rest
local function extract_kb_only_relations(ep_data)
    local col_table, row_table, col_seq_table, row_seq_table = {}, {}, {}, {}
    for i = 1, ep_data.rel:size(2) do
        local col, row, col_seq, row_seq = extract_example(ep_data, i)

        local mask = torch.ByteTensor(row:size(1), 1):fill(0)
        -- get indices in the row tensor that match any of our kb ids
        for _, kb_id in pairs(kb_rels) do
            mask:add(row:eq(kb_id:expandAs(row)))
        end
        local index = {}
        for idx = 1, mask:size(1) do if mask[idx][1] == 1 then table.insert(index, idx) end end

        if mask:sum() > 0 then
            table.insert(col_table, col:index(1, torch.LongTensor(index)))
            table.insert(col_seq_table, col_seq:index(1, torch.LongTensor(index)))
            table.insert(row_table, row:index(1, torch.LongTensor(index)))
            table.insert(row_seq_table, row_seq:index(1, torch.LongTensor(index)))
        end
    end
    return col_table, row_table, col_seq_table, row_seq_table
end

local i = 0
for num_relations, ep_data in pairs(original_data) do
--    num_relations = 131
--    ep_data = original_data[num_relations]
    io.write('\rProcessing : '..i); io.flush(); i = i + 1
    if (type(ep_data) == 'table' and i > 1) then
        local col_table, row_table, col_seq_table, row_seq_table
        if params.kbOnly then col_table, row_table, col_seq_table, row_seq_table = extract_kb_only_relations(ep_data)
        else col_table, row_table, col_seq_table, row_seq_table =  extract_relations(ep_data) end

        reformatted_data[num_relations] = {
            row_seq = nn.JoinTable(1)(row_seq_table), row = nn.JoinTable(1)(row_table),
            col_seq = nn.JoinTable(1)(col_seq_table), col = nn.JoinTable(1)(col_table) }
        reformatted_data[num_relations].count = reformatted_data[num_relations].row:size(1)
    end
end
print('\nDone')
reformatted_data.max_length = i

-- SAVE FILE
torch.save(params.outFile, reformatted_data)


