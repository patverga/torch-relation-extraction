--
-- User: pat
-- Date: 10/20/15
--

local SelectLast, parent = torch.class('nn.SelectLast', 'nn.Module')

function SelectLast:__init(dimension)
    parent.__init(self)
    self.dimension = dimension
end

function SelectLast:updateOutput(input)
    local index = input:size(self.dimension)
    local output = input:select(self.dimension,index);
    self.output:resizeAs(output)
    return self.output:copy(output)
end

function SelectLast:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    local index = input:size(self.dimension)
    self.gradInput:select(self.dimension,index):copy(gradOutput)
    return self.gradInput
end