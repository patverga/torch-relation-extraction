
--
-- User: pat
-- Date: 2/17/16
--
local MaxOneHot, parent = torch.class('nn.MaxOneHot', 'nn.Max')

function MaxOneHot:__init(dimension, nInputDims)
    parent.__init(self, dimension, nInputDims)
end

function MaxOneHot:updateOutput(input)
    self:_lazyInit()
    local dimension = self:_getPositiveDimension(input)
    torch.max(self._max, self._indices, input, dimension)
    self.output = self.output:resizeAs(input):zero():scatter(dimension, self._indices, self._max)
    return self.output
end

function MaxOneHot:_lazyInit()
    self._max = self._max or self.output.new()
    self._indices = self._indices or
            (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
    self.gradInput = self._max.new()
end

function MaxOneHot:updateGradInput(input, gradOutput)
    self:_lazyInit()
    self.gradInput = gradOutput
    return self.gradInput
end