--
-- User: pat
-- Date: 4/14/16
--

local SelectTable_hacked, parent = torch.class('nn.SelectTable_hacked', 'nn.SelectTable')


local function zeroTableCopy(t1, t2)
    for k, v in pairs(t2) do
        if (torch.type(v) == "table") then
            t1[k] = zeroTableCopy(t1[k] or {}, t2[k])
        else
            if not t1[k] then
                t1[k] = v:clone():zero()
            else
                t1[k]:resizeAs(v)
                t1[k]:zero()
            end
        end
    end
    for k, v in pairs(t1) do
        if not t2[k] then
            t1[k] = nil
        end
    end
    return t1
end


function SelectTable_hacked:updateGradInput(input, gradOutput)
    -- make gradInput a zeroed copy of input
    zeroTableCopy(self.gradInput, input)
    -- handle negative indices
    local index = self.index < 0 and #input + self.index + 1 or self.index
    -- copy into gradInput[index] (necessary for variable sized inputs)
    assert(self.gradInput[index])
    if gradOutput then
        nn.utils.recursiveCopy(self.gradInput[index], gradOutput)
    end

    return self.gradInput
end

