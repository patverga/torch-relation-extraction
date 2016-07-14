--
-- User: pat
-- Date: 4/14/16
--

local JoinTable_hacked, parent = torch.class('nn.JoinTable_hacked', 'nn.JoinTable')


function JoinTable_hacked:updateGradInput(input, gradOutput)
    local dimension = self:_getPositiveDimension(input)

    if gradOutput and gradOutput:dim() > 0 then
        for i=1,#input do
            if self.gradInput[i] == nil then
                self.gradInput[i] = input[i].new()
            end
            self.gradInput[i]:resizeAs(input[i])
        end

        -- clear out invalid gradInputs
        for i=#input+1, #self.gradInput do
            self.gradInput[i] = nil
        end

        local offset = 1
        for i=1,#input do
            local currentOutput = input[i]
            local currentGradInput = gradOutput:narrow(dimension, offset,
                currentOutput:size(dimension))
            self.gradInput[i]:copy(currentGradInput)
            offset = offset + currentOutput:size(dimension)
        end
    else
        self.gradInput = input
    end
    return self.gradInput
end

