--
-- User: pat
-- Date: 9/17/15
--


require 'torch'

--[[
    Takes a tac candidate file and exports a candidtate file to outfile with subtrings between entities extracted
]]--
local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input candidate file')
cmd:option('-outFile', '', 'scored candidate out file')
cmd:option('-delim', ' ', 'delimiter to break string on')

local params = cmd:parse(arg)

--- process a single line from a candidate file ---
local function process_line(line)
    local out_line
    local e1, e2, rel, label = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    local tokens = {}
    local rel_string = ''
    for token in string.gmatch(rel, "[^" .. params.delim .. "]+") do
        table.insert(tokens, token)
    end
    for i = 1, #tokens do
        if #tokens <= 6 or i <= 3 or i > #tokens -3 then
            rel_string = rel_string .. tokens[i]
            if i < #tokens then rel_string = rel_string .. ' ' end
        elseif i == 4 then
            rel_string = rel_string .. '[' .. math.floor(torch.log(#tokens - 6)/torch.log(2)) .. ']' .. ' '
        end
    end
    local out_line = e1 .. '\t' .. e2 .. '\t' .. rel_string .. '\t' .. label
    return out_line
end

--- process the candidate file and convert to torch ---
local function process_file()
    local line_num = 0
    local out_file = io.open(params.outFile, "w")
    print('Processing data')
    for line in io.lines(params.inFile) do
        local out_line = process_line(line)
        out_file:write(out_line .. '\n')
        line_num = line_num + 1
        if line_num % 10000 == 0 then io.write('\rline : ' .. line_num); io.flush() end
    end
    print ('\rProcessed ' .. line_num .. ' lines')
end

---- main
process_file()
