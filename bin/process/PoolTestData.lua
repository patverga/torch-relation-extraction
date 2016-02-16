--
-- User: pat
-- Date: 2/10/16
--

require 'nn'

local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-keyFile', '', 'file that maps entity pairs to relations')
cmd:option('-outFile', '', 'out file')
local params = cmd:parse(arg)


-- load key file int map
print('Loading ep-rel map from ' .. params.keyFile)
local key = torch.load(params.keyFile)
local ep_rel_map = {}
local ep_seq_map = {}
for a, dat in pairs(key) do
    if dat and torch.type(dat) == 'table' and dat.ep then
        for i = 1, dat.ep:size(1) do
            ep_rel_map[dat.ep[i]] = dat.rel:narrow(1,i,1)
            ep_seq_map[dat.ep[i]] = dat.seq:narrow(1,i,1)
        end
    end
end

print('Converting data from ' .. params.inFile .. ' and exporting to ' .. params.outFile)
local data = torch.load(params.inFile)
local mapped_data = {}
local missing = 0
for _, sub_data in pairs(data) do
    if torch.type(sub_data) == 'table' then
        for i = 1, sub_data.row:size(1) do
            local ep = sub_data.row[i][1]
            local rels = ep_rel_map[ep]
            local seq = ep_seq_map[ep]
            if rels then
                local k = rels:size(2)
                local label = sub_data.label[i]
                local kb_rel = sub_data.col[i]
                local kb_rel_seq = sub_data.col_seq[i]
                if not mapped_data[k] then mapped_data[k] = {row={}, row_seq ={}, col={}, col_seq={}, label ={}} end
                table.insert(mapped_data[k].col, rels)
                table.insert(mapped_data[k].col_seq, seq)
                table.insert(mapped_data[k].label, label)
                table.insert(mapped_data[k].row, kb_rel)
                table.insert(mapped_data[k].row_seq, kb_rel_seq)
            else
                missing = missing + 1
            end
        end
    end
end
print('couldnt map ' .. missing .. ' entity pairs')

for len, sub_data in pairs(mapped_data) do
    mapped_data[len].label = nn.JoinTable(1)(sub_data.label)
    mapped_data[len].col = nn.JoinTable(1)(sub_data.col)
    mapped_data[len].col_seq = nn.JoinTable(1)(sub_data.col_seq)
    mapped_data[len].row = nn.JoinTable(1)(sub_data.row)
    -- todo, something seems wrong with row_seq
    mapped_data[len].row_seq = nn.JoinTable(1)(sub_data.row_seq)
    mapped_data[len].count = sub_data.label:size(1)
end
-- SAVE FILE
torch.save(params.outFile, mapped_data)
print('Done')




