--
-- User: pat
-- Date: 4/13/16
--

-- Randomly generate a tensor of the same size as input with values in the same range as those along specified dimension

local RandomTensor, parent = torch.class('nn.RandomTensor', 'nn.Module')

function RandomTensor:__init(dimension, p)
    self.p = p or 1.0
    self.p_decay = 0 --.0001
    self.dim = dimension
    self.train = true
    parent.__init(self)
end

function RandomTensor:updateOutput(input)
    if self.p > 0 then self.p = self.p - self.p_decay end
    self.output:resizeAs(input)
    if self.train and self.p > 0 and torch.rand(1)[1] <= self.p then
        local max_val = input:max(self.dim):view(-1,1)
        local min_val = input:min(self.dim):view(-1,1)
        self.output:copy(torch.rand(input:size())):cmul((max_val-min_val):expandAs(input)):add(-min_val:expandAs(input))
    else
        self.output:fill(0)
    end
    return self.output
end


function RandomTensor:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:fill(0)
    return self.gradInput
end

function RandomTensor:clearState()
    -- don't call set because it might reset referenced tensors
    local function clear(f)
        if self[f] then
            if torch.isTensor(self[f]) then
                self[f] = self[f].new()
            elseif type(self[f]) == 'table' then
                self[f] = {}
            else
                self[f] = nil
            end
        end
    end
    clear('output')
    clear('gradInput')
    return self
end