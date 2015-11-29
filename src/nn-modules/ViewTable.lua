--
-- User: pat
-- Date: 9/9/15
--

local ViewTable, parent = torch.class('nn.ViewTable', 'nn.View')

--function ViewTable:__init(...)
--    parent.__init(self)
--    if select('#', ...) == 1 and torch.typename(select(1, ...)) == 'torch.LongStorage' then
--        self.size = select(1, ...)
--    else
--        self.size = torch.LongStorage({...})
--    end
--
--    self.numElements = 1
--    local inferdim = false
--    for i = 1,#self.size do
--        local szi = self.size[i]
--        if szi >= 0 then
--            self.numElements = self.numElements * self.size[i]
--        else
--            assert(szi == -1, 'size should be positive or -1')
--            assert(not inferdim, 'only one dimension can be at -1')
--            inferdim = true
--        end
--    end
--
--    self.output = nil
--    self.gradInput = nil
--    self.numInputDims = nil
--end
--
--function ViewTable:setNumInputDims(numInputDims)
--    self.numInputDims = numInputDims
--    return self
--end

local function batchsize(input, size, numInputDims, numElements)
    local ind = input:nDimension()
    local isz = input:size()
    local maxdim = numInputDims and numInputDims or ind
    local ine = 1
    for i=ind,ind-maxdim+1,-1 do
        ine = ine * isz[i]
    end

    if ine % numElements ~= 0 then
        error(string.format(
            'input view (%s) and desired view (%s) do not match',
            table.concat(input:size():totable(), 'x'),
            table.concat(size:totable(), 'x')))
    end

    -- the remainder is either the batch...
    local bsz = ine / numElements

    -- ... or the missing size dim
    for i=1,size:size() do
        if size[i] == -1 then
            bsz = 1
            break
        end
    end

    -- for dim over maxdim, it is definitively the batch
    for i=ind-maxdim,1,-1 do
        bsz = bsz * isz[i]
    end

    -- special card
    if bsz == 1 and (not numInputDims or input:nDimension() <= numInputDims) then
        return
    end

    return bsz
end

function ViewTable:updateOutput(input)
    local bsz = batchsize(input[1], self.size, self.numInputDims, self.numElements)
    self.output = {}
    for i = 1,#input do
        if bsz then
            self.output[i] = input[i]:view(bsz, table.unpack(self.size:totable()))
        else
            self.output[i] = input[i]:view(self.size)
        end
    end
    return self.output
end

function ViewTable:updateGradInput(input, gradOutput)
    self.gradInput = {}
    for i = 1, #input do
        self.gradInput[i] = gradOutput[i]:view(input[i]:size())
    end

    return self.gradInput
end