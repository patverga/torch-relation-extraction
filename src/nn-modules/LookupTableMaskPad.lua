--
-- User: pv
-- Date: 3/25/16
--

local LookupTableMaskPad, parent = torch.class('nn.LookupTableMaskPad', 'nn.LookupTable')

function LookupTableMaskPad:__init(nIndex, nOutput, padIdx, max_norm)
    parent.__init(self, nIndex, nOutput, padIdx, max_norm)
    self.padIdx = padIdx
end

function LookupTableMaskPad:updateOutput(input)
    self.weight[self.padIdx]:zero()
    return parent.updateOutput(self, input)
end

