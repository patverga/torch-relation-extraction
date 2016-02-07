--
-- User: pat
-- Date: 1/15/16
--



local BPRLoss, parent = torch.class('nn.BPRLoss', 'nn.Criterion')

function BPRLoss:__init()
    parent.__init(self)
    self.output = nil
end

function BPRLoss:updateOutput(input, y)
    local theta = input[1] - input[2]
    self.output = self.output and self.output:resizeAs(theta) or theta:clone()
    self.output = self.output:fill(1):cdiv(torch.exp(-theta):add(1))
    local err = torch.log(self.output):mean()
    return err
end

function BPRLoss:updateGradInput(input, y)
    local step = self.output:mul(-1):add(1)
    self.gradInput = { -step, step }
    return self.gradInput
end
