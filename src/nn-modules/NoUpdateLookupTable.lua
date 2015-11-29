--
-- User: pat
-- Date: 10/22/15
--

local NoUpdateLookupTable, parent = torch.class('nn.NoUpdateLookupTable', 'nn.LookupTable')

function NoUpdateLookupTable:accGradParameters(input, gradOutput, scale)
end

function NoUpdateLookupTable:updateParameters(learningRate)
end