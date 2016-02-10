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
for a, dat in pairs(key) do
    if dat and torch.type(dat) == 'table' and dat.ep then
        for i = 1, dat.ep:size(1) do
            ep_rel_map[dat.ep[i]] = dat.seq:narrow(1,i,1)
        end
    end
end

print('Converting data from ' .. params.inFile .. ' and exporting to ' .. params.outFile)
local data = torch.load(params.inFile)[1]
local mapped_data = {}
local missing = 0
for i = 1, data.ep:size(1) do
    local ep = data.ep[i][1]
    local rels = ep_rel_map[ep]
    if rels then
        local seq = rels:size(2)
        local label = data.label[i]
        local kb_rel = data.rel[i]
        if not mapped_data[seq] then mapped_data[seq] = {col_seq={}, label ={}, row_seq ={}} end
--        table.insert(mapped_data[seq].ep, ep)
        table.insert(mapped_data[seq].col_seq, rels)
        table.insert(mapped_data[seq].label, label)
        table.insert(mapped_data[seq].row_seq, kb_rel)
    else
        missing = missing + 1
    end
end
print('couldnt map ' .. missing .. ' entity pairs')

for len, data in pairs(mapped_data) do
    mapped_data[len].label = nn.JoinTable(1)(data.label)
    mapped_data[len].label = mapped_data[len].label:view(mapped_data[len].label:size(1), 1)
    mapped_data[len].col_seq = nn.JoinTable(1)(data.col_seq)
    mapped_data[len].row_seq = nn.JoinTable(1)(data.row_seq)
    mapped_data[len].row_seq = mapped_data[len].row_seq:view(mapped_data[len].row_seq:size(1), 1)
    mapped_data[len].row = torch.Tensor(mapped_data[len].row_seq:size(1))
    mapped_data[len].col = torch.Tensor(mapped_data[len].col_seq:size(1))
end



-- SAVE FILE
torch.save(params.outFile, mapped_data)
print('Done')




