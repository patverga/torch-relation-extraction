--
-- User: pat
-- Date: 9/17/15
--


require 'torch'

--[[
    Takes a tac candidate file and exports a candidtate file to outfile with subtrings between entities extracted
]]--
local cmd = torch.CmdLine()
cmd:option('-candidates', '', 'input candidate file')
cmd:option('-outFile', '', 'scored candidate out file')
cmd:option('-delim', ' ', 'delimiter to break string on')
cmd:option('-logRelations', false, 'Use log relation vectors instead of tokens')

local params = cmd:parse(arg)

--- process a single line from a candidate file ---
local function process_line(line)
    local query_id, tac_rel, sf_2, doc_info, start_1, end_1, start_2, end_2, pattern_rel
    = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")

    -- we only want tokens between the two args
    local start_idx = tonumber(end_1)
    local end_idx = tonumber(start_2)
    local arg1_first = true
    if (start_idx > end_idx) then
        start_idx, end_idx, arg1_first = tonumber(end_2), tonumber(start_1), false
    end
    local idx = 0

    local tokens = {}
    for token in string.gmatch(pattern_rel, "[^" .. params.delim .. "]+") do
        if idx >= start_idx and idx < end_idx then table.insert(tokens, token) end
        idx = idx + 1
    end

    local rel_string = arg1_first and "$ARG1 " or "$ARG2 "
    for i = 1, #tokens do
        if not params.logRelations or #tokens <= 4 or i <= 2 or i > #tokens -2 then
            rel_string = rel_string .. tokens[i] .. ' '
        elseif i == 3 then
            rel_string = rel_string .. '[' .. math.floor(torch.log(#tokens - 4)/torch.log(2)) .. ']' .. ' '
        end
    end
    rel_string = rel_string .. (arg1_first and "$ARG2" or "$ARG1")
    local rel_string = arg1_first and "$ARG1 " or "$ARG2 "
    local idx = 0
    for token in string.gmatch(pattern_rel, "[^" .. params.delim .. "]+") do
        if idx >= start_idx and idx < end_idx then
            rel_string = rel_string .. token .. ' '
        end
        idx = idx + 1
    end
    rel_string = rel_string .. (arg1_first and "$ARG2" or "$ARG1")
    local out_line = query_id .. '\t' .. tac_rel .. '\t' .. sf_2 .. '\t' .. doc_info .. '\t'
            .. start_1 .. '\t' .. end_1 .. '\t' .. start_2 .. '\t' .. end_2 .. '\t' .. rel_string

    return out_line
end

--- process the candidate file and convert to torch ---
local function process_file()
    local line_num = 0
    local out_file = io.open(params.outFile, "w")
    print('Processing data')
    for line in io.lines(params.candidates) do
        local out_line = process_line(line)
        out_file:write(out_line .. '\n')
        line_num = line_num + 1
        if line_num % 10000 == 0 then io.write('\rline : ' .. line_num); io.flush() end
    end
    print ('\rProcessed ' .. line_num .. ' lines')
end

---- main
process_file()
