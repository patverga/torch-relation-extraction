--
-- User: pv
-- Date: 4/11/16
--

local PatternDropout, parent = torch.class('nn.PatternDropout', 'nn.Module')


function PatternDropout:__init(max_patterns)
    parent.__init(self)
    self.max_patterns = max_patterns
    self.train = true
end


function PatternDropout:updateOutput(input)
    if not self.train or input:size(2) <= self.max_patterns then
        self.output = input
    else
        self.rand_indices = torch.rand(self.max_patterns):mul(input:size(2)):floor():add(1)
        self.output = input:index(2, self.rand_indices:long())
    end
    return self.output
end


function PatternDropout:updateGradInput(input, gradOutput)
    self.gradInput = input
    return self.gradInput
end