--
-- User: pat
-- Date: 3/1/16
--
require 'torch'

--[[
--  Each line of input file should be [key \t v1,v2,...,vn] with an optional delimeter character for list seperator
-- All values must be int mapped
 - output will be table where t[key]=tensor({v1,v2,...,vn})
-- ]]--

local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input candidate file')
cmd:option('-outFile', '', 'scored candidate out file')
cmd:option('-delim', ',', 'delimiter to break string on')

local params = cmd:parse(arg)

local map_table = {}
io.write('Loading map... '); io.flush()
for line in io.lines(params.inFile) do
    local key, values = string.match(line, "([^\t]+)\t([^\t]+)")
    local value_table = {}
    if key and values then
        for v in string.gmatch(values, "[^" .. params.delim .. "]+") do table.insert(value_table, tonumber(v)) end
        map_table[tonumber(key)] = torch.Tensor(value_table)
    end
end

torch.save(params.outFile, map_table)
print('Done')
