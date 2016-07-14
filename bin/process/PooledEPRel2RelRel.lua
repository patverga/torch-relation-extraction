require 'nn'

--[[
    Given a file processed by IntFile2PoolRelationsTorch.lua mapping entity pairs to a set of relations
    creates a new file mapping a single relation to a set of relations
  ]]--

local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-outFile', '', 'out file')
cmd:option('-relationSubset', '', 'only take out relations in this map file')
cmd:option('-maxSamples', 0, 'maximum number of samples to take for a given entity pair - 0 = all')
cmd:option('-maxColumns', 0, 'maximum number of columns to use for a single example - 0 = all')
cmd:option('-dummyRelation', 0, 'if > 0, add dummy relation with this index to every row')
local params = cmd:parse(arg)


local relation_subset = {}
if params.relationSubset ~= '' then
    io.write('Loading kb map... ')
    for line in io.lines(params.relationSubset) do
        local token, id = string.match(line, "([^\t]+)\t([^\t]+)")
        if token and id then
            id = tonumber(id)
            if id > 1 then table.insert(relation_subset, torch.Tensor(1,1):fill(id)) end
        end
    end
    print('Done')
end

-- DATA
print('loading file: ', params.inFile)
local original_data = torch.load(params.inFile)
-- prepare new data object dNew
local reformatted_data = {num_cols = math.max(params.dummyRelation, original_data.num_rels),
    num_rows = math.max(params.dummyRelation, original_data.num_rels),
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

    if  params.maxColumns > 0 and col:size(2) > params.maxColumns then
        local rand_subset = torch.randperm(col:size(2)):narrow(1, 1, math.min(col:size(2), params.maxSamples)):long()
        col = col:index(2,rand_subset)
        col_seq = col_seq:index(2,rand_subset)
    end

    return col, row, col_seq, row_seq
end

-- for each r in the relation set, create #r examples - take one out and leave the rest
local function extract_relations(ep_data)
    local col_table, row_table, col_seq_table, row_seq_table = {}, {}, {}, {}
    local shuffle = torch.randperm(ep_data.rel:size(2))
    if params.maxSamples > 0 then
        shuffle = shuffle:narrow(1, 1, math.min(shuffle:size(1), params.maxSamples))
    end
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
        for _, kb_id in pairs(relation_subset) do
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
    io.write('\rProcessing : '..i); io.flush(); i = i + 1
    if (type(ep_data) == 'table' and i > 1) then
        local col_table, row_table, col_seq_table, row_seq_table
        if params.relationSubset ~= '' then col_table, row_table, col_seq_table, row_seq_table = extract_kb_only_relations(ep_data)
        else col_table, row_table, col_seq_table, row_seq_table =  extract_relations(ep_data) end
        if #row_table > 0 then
            local row, col = nn.JoinTable(1)(row_table), nn.JoinTable(1)(col_table)
            if params.dummyRelation > 0 then col = col:cat(row:clone():fill(params.dummyRelation):view(-1,1)) end
            if params.maxColumns == 0 or num_relations <= params.maxColumns or not reformatted_data[params.maxColumns] then
                reformatted_data[num_relations-1] = {
                    row_seq = nn.JoinTable(1)(row_seq_table), row = row,
                    col_seq = nn.JoinTable(1)(col_seq_table), col = col,
                    count = row:size(1)
                }
            else
                reformatted_data[params.maxColumns].row_seq = reformatted_data[params.maxColumns].row_seq:cat(nn.JoinTable(1)(row_seq_table), 1)
                reformatted_data[params.maxColumns].row = reformatted_data[params.maxColumns].row:cat(row, 1)
                reformatted_data[params.maxColumns].col_seq = reformatted_data[params.maxColumns].col_seq:cat(nn.JoinTable(1)(col_seq_table), 1)
                reformatted_data[params.maxColumns].col = reformatted_data[params.maxColumns].col:cat(col, 1)
                reformatted_data[params.maxColumns].count = reformatted_data[params.maxColumns].count + row:size(1)
            end
        end
    end
end
print('\nSaving')
reformatted_data.max_length = i

-- SAVE FILE
torch.save(params.outFile, reformatted_data)
print('Done')


