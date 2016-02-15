local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-outFile', '', 'out file')
local params = cmd:parse(arg)

-- DATA
print('loading file: ', params.inFile)
local original_data = torch.load(params.inFile)
-- prepare new data object dNew
local reformatted_data = {num_cols = original_data.num_rels, num_rows = original_data.num_eps,
    num_col_tokens = original_data.num_tokens, num_row_tokens = original_data.num_tokens}


-- LOOP OVER KEYS TO RESTRUCTURE VALUES
local i = 0
for num_relations, ep_data in pairs(original_data) do
    io.write('\rProcessing : '..i); io.flush()
    i = i + 1

    -- other cases: several relations
    if (type(ep_data) == 'table' and i > 1) then
        local col, row, col_seq, row_seq
        -- LOOP OVER RELATIONS TO RESTRUCTURE THEM
        for s = 1, ep_data.rel:size(2) do
            -- create new RelTest in ep from ONE dimension of seq
            row = ep_data.rel:select(2, s)
            row_seq = ep_data.seq:select(2, s)
            local range
            if s == 1 then
                range = torch.range(s+1, ep_data.rel:size(2))
            elseif s == ep_data.rel:size(2) then
                range = torch.range(1, ep_data.rel:size(2)-1)
            else
                range = torch.range(1, s-1):cat(torch.range(s, ep_data.rel:size(2)))
            end
            col = ep_data.rel:index(2, range:long())
            col = col:view(col:size(1), col:size(2))
            col_seq = ep_data.seq:index(2, range:long())

        end
        reformatted_data[num_relations] = {row_seq = row_seq, row = row, count = row_seq:size(1), col_seq = col_seq, col = col }
    end
end
print('\nDone')
reformatted_data.max_length = i

-- SAVE FILE
torch.save(params.outFile, reformatted_data)


