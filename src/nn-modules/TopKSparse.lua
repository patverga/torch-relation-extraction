--
-- User: pat
-- Date: 4/27/16
--


local TopKSparse, parent = torch.class('nn.TopKSparse', 'nn.Max')



function TopKSparse:__init(K, dimension, nInputDims)
    parent.__init(self, dimension, nInputDims)
    self.K = K
end

function TopKSparse:updateOutput(input)
    self:_lazyInit()
    local dimension = self:_getPositiveDimension(input)
    local k = math.min (self.K, input:size(dimension))
    torch.topk(self._topk, self._indices, input, k, dimension, true)
    self.output = self.output:resizeAs(input):zero():scatter(dimension, self._indices, self._topk)
    return self.output
end

function TopKSparse:_lazyInit()
    self._topk = self._topk or self.output.new()
    self._indices = self._indices or
            (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
    self.gradInput = self._topk.new()
end

function TopKSparse:updateGradInput(input, gradOutput)
    self:_lazyInit()
    self.gradInput = gradOutput
    return self.gradInput
end