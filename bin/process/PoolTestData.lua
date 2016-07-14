--
-- User: pat
-- Date: 2/10/16
--

require 'nn'

local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-inDir', '', 'input directory filled with input files - if this is given, outFile is assumed to be a direcotry')
cmd:option('-keyFile', '', 'file that maps entity pairs to relations')
cmd:option('-outFile', '', 'out file')
local params = cmd:parse(arg)


-- load key file int map
print('Loading ep-rel map from ' .. params.keyFile)
local key = torch.load(params.keyFile)
local ep_rel_map = {}
local ep_seq_map = {}
for _, dat in pairs(key) do
    if dat and torch.type(dat) == 'table' and dat.ep then
        for i = 1, dat.ep:size(1) do
            ep_rel_map[dat.ep[i]] = dat.rel:narrow(1,i,1)
            ep_seq_map[dat.ep[i]] = dat.seq:narrow(1,i,1)
        end
    end
end

local function process_file(in_file, out_file)
    print('Converting data from ' .. in_file .. ' and exporting to ' .. out_file)
    local data = torch.load(in_file)
    local missing = 0
    local mapped = 0
    local out_data = {}
    for _, sub_data in pairs(data) do
        local mapped_data = {}
        if torch.type(sub_data) == 'table' then
            local length = sub_data.row and sub_data.row:size(1) or sub_data.ep:size(1)
            for i = 1, length do
                local ep = sub_data.row and sub_data.row[i][1] or sub_data.ep[i][1]
                local rels = ep_rel_map[ep]
                local seq = ep_seq_map[ep]
                if rels then
                    local k = rels:size(2)
                    local label = sub_data.label[i]
                    local kb_rel = sub_data.col and sub_data.col[i] or sub_data.rel[i]
                    local kb_rel_seq = sub_data.col_seq and sub_data.col_seq[i] or sub_data.seq[i]
                    if not mapped_data[k] then mapped_data[k] = {row={}, row_seq ={}, col={}, col_seq={}, label ={}} end
                    table.insert(mapped_data[k].col, rels)
                    table.insert(mapped_data[k].col_seq, seq)
                    table.insert(mapped_data[k].label, label)
                    table.insert(mapped_data[k].row, kb_rel)
                    table.insert(mapped_data[k].row_seq, kb_rel_seq)
                    mapped = mapped + 1
                else
                    missing = missing + 1
                end
            end
        end
        for len, sub_data in pairs(mapped_data) do
            mapped_data[len].label = nn.JoinTable(1)(sub_data.label)
            mapped_data[len].col = nn.JoinTable(1)(sub_data.col)
            mapped_data[len].col_seq = nn.JoinTable(1)(sub_data.col_seq)
            mapped_data[len].row = nn.JoinTable(1)(sub_data.row)
            mapped_data[len].row_seq = nn.JoinTable(1)(sub_data.row_seq)
            mapped_data[len].count = sub_data.label:size(1)
        end
        table.insert(out_data, mapped_data)
    end

    -- SAVE FILE
    torch.save(out_file, out_data)
    print('Done. Mapped : ' .. mapped .. ' entity pairs. \nCouldnt map : ' .. missing .. ' entity pairs')
end


if params.inFile ~= '' then
    process_file(params.inFile, params.outFile)
elseif params.inDir ~= '' then
    for file in io.popen('ls ' .. params.inDir):lines() do
        print ('Processing ' .. file)
        process_file(params.inDir..'/'..file, params.outFile..'/'..file)
    end
else
    print('Must supply either -inFile or -inDir')
end

