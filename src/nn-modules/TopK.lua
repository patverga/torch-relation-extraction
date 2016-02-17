--
-- User: pat
-- Date: 2/17/16
--
local TopK, parent = torch.class('nn.TopK', 'nn.Max')

function TopK:__init(K, dimension, nInputDims)
    parent:__init(dimension, nInputDims)
    self.K = K
end


function TopK:updateOutput(input)
    self:_lazyInit()
    local dimension = self:_getPositiveDimension(input)
    local k = math.min (self.K, input:size(2))
    self._output, self._indices = torch.topk(input, k, dimension, true)
    self.output = self._output
    return self.output
end

function TopK:_lazyInit()
    parent:_lazyInit()
    self.gradInput = self._output.new()
end

function TopK:updateGradInput(input, gradOutput)
    self:_lazyInit()
    local dimension = self:_getPositiveDimension(input)
    local gradOutputView
    gradOutputView = gradOutput
    self.gradInput:resizeAs(input):zero():scatter(dimension, self._indices, gradOutputView)
    return self.gradInput
end

