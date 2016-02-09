--[[
This code is to restructure data.
Input: 	data d stores tensor of entity-pairs observed with relations (consists out of (key, value)-pairs)
	key k with k=3: indicates number of relations that entity-pairs were observed with (d[k] with k=3 stores all entity-pairs observed with any 3 relations)
	value v:	ep : DoubleTensor - size: 6213		--> entity-pairs
	  		count : 0
	  		seq : DoubleTensor - size: 6213x3x50	--> entity-pairs x relations x embedding with padding (???)
	  		num_eps : 55341
	  		num_tokens : 62837
Output: data dNew contains same data d restructured with all combinations of extracting ONE relation as test-relation at a time
	key k with k=3: indicates number of relations that entity-pairs were observed with (d[k] with k=3 stores all entity-pairs observed with any 3 relations)
	value v:	ep : DoubleTensor - size: 18639x1x50	--> (6213 entity-pairs x 3 relations) x 1 test-relations x embedding with padding (???)
	  		count : 0
	  		seq : DoubleTensor - size: 18639x2x50	--> (6213 entity-pairs x 3 relations) x 2 observed-relations x embedding with padding (???)
	  		num_eps : 55341
	  		num_tokens : 62837
]] --

print('restructureData start')
local cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-outFile', '', 'out file')
local params = cmd:parse(arg)

-- DATA
print('loading file: ', params.inFile)
local d = torch.load(params.inFile)
-- prepare new data object dNew
local dNew = {num_cols=1}


-- LOOP OVER KEYS TO RESTRUCTURE VALUES
local i = 0
for k, v in pairs(d) do
    io.write('\rProcessing : '..i); io.flush()
    i = i + 1

    -- special case: only one relation
    --if(i == 1) then
    --	dNew[k] = v
    --end

    -- other cases: several relations
    if (type(v) == 'table' and i > 1) then
        local yDim_original = v.seq:size()[1]
        local xDim_original = v.seq:size()[2]
        local zDim_original = v.seq:size()[3]
        local yDim_row_seq = yDim_original * xDim_original
        -- print("yDim: ", yDim_row_seq)
        local xDim_row_seq = 1
        local zDim_row_seq = zDim_original
        local yDim_seq = yDim_row_seq
        local xDim_seq = xDim_original - 1
        local zDim_seq = zDim_original

        local row_seq = torch.Tensor(yDim_row_seq, xDim_row_seq, zDim_row_seq)
        local col_seq = torch.Tensor(yDim_seq, xDim_seq, zDim_seq)


        -- LOOP OVER RELATIONS TO RESTRUCTURE THEM
        for s = 1, xDim_original do
            -- print("s: ", s)
            local ySI_target = ((s - 1) * yDim_original) + 1
            local yEI_target = ySI_target + yDim_original - 1
            -- print("y start index", ySI_target)
            -- print("y end index", yEI_target)

            -- create new RelTest in ep from ONE dimension of seq
            row_seq[{ { ySI_target, yEI_target }, { 1 }, { 1, zDim_original } }] = torch.Tensor(v.seq[{ { 1, yDim_original }, { s }, { 1, zDim_original } }])


            -- create new RelObs in seq from ALL OTHER dimension of seq
            -- special case: test-relation is the FIRST
            if (s == 1) then
                -- print("s == 1")
                col_seq[{ { ySI_target, yEI_target }, {}, { 1, zDim_original } }] = torch.Tensor(v.seq[{ { 1, yDim_original }, { 2, xDim_seq + 1 }, { 1, zDim_original } }])

                -- special case: test-relation is the LAST
            elseif (s == (xDim_seq + 1)) then
                -- print("s == ",xDim_seq + 1)
                col_seq[{ { ySI_target, yEI_target }, {}, { 1, zDim_original } }] = torch.Tensor(v.seq[{ { 1, yDim_original }, { 1, xDim_seq }, { 1, zDim_original } }])

                -- other cases: test-relation is the somewher in the MIDDLE
            else
                -- print("1 < s <", (xDim_seq + 1))
                local left = torch.Tensor(v.seq[{ { 1, yDim_original }, { 1, s - 1 }, { 1, zDim_original } }])
                -- print(left:size())
                local right = torch.Tensor(v.seq[{ { 1, yDim_original }, { s + 1, xDim_seq + 1 }, { 1, zDim_original } }])
                -- print(right:size())
                col_seq[{ { ySI_target, yEI_target }, {}, { 1, zDim_original } }] = left:cat(right, 2)
            end
        end

        -- store new structure
        local newV = {}
        newV.row_seq = row_seq:view(row_seq:size(1), row_seq:size(3)) -- new RelTest in ep from ONE dimension of seq
        newV.row = torch.Tensor(row_seq:size(1)) -- new RelTest in ep from ONE dimension of seq
        newV.count = row_seq:size(1)
        newV.col_seq = col_seq -- new RelObs in seq from ALL OTHER dimension of seq
        dNew.num_col_tokens = v.num_tokens
        dNew.num_row_tokens = v.num_tokens
        newV.num_eps = v.num_eps

        dNew[k] = newV
    end
end
dNew.max_length = i
print(dNew)

-- SAVE FILE
torch.save(params.outFile, dNew)

print('restructureData end')


