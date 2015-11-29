--
-- User: pat
-- Date: 11/4/15
--

local WordDropout, Parent = torch.class('nn.WordDropout', 'nn.Module')

function WordDropout:__init(p, unkIndex)
    Parent.__init(self)
    self.p = p or 0.5
    self.unkIndex = unkIndex
    self.train = true
    if self.p >= 1 or self.p < 0 then
        error('<Dropout> illegal percentage, must be 0 <= p < 1')
    end
    self.noise = torch.Tensor()
end

function WordDropout:updateOutput(input)
    self.output:resizeAs(input):copy(input)
    if self.train then
        self.noise:resizeAs(input)
        self.noise:bernoulli(1 - self.p)
        self.output:cmul(self.noise)
        self.output:add(self.unkIndex, self.noise:eq(0):typeAs(self.output))
    end
    return self.output
end

function WordDropout:updateGradInput(input, gradOutput)
end

function WordDropout:setp(p)
    self.p = p
end

function WordDropout:__tostring__()
    return string.format('%s(%f)', torch.type(self), self.p)
end
