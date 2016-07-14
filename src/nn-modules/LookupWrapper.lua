--
-- User: pat
-- Date: 5/23/16
--
local LookupWrapper, parent = torch.class('nn.LookupWrapper', 'nn.Container')

function LookupWrapper:__init(pos_net, null_table, skip_value, dim)
    parent.__init(self)
    table.insert(self.modules, pos_net)
    table.insert(self.modules, null_table)
    self.pos_net = pos_net
    self.null_table = null_table
    self.skip_value = skip_value
    self.dim = dim
end

function LookupWrapper:updateOutput(input)
    -- get embeddings directly for null tokens

    -- determine the null mask by looking at the int value of the first embedding
    -- skip_value should be the int value that indicates <null>
    self.null_mask = input:le(self.skip_value)
    self.num_nulls = self.null_mask:sum()
    if(self.num_nulls > 0) then
        -- get the things we want to put through the null table
        self.null_input = input:maskedSelect(self.null_mask:repeatTensor(1,input:size(2))):reshape(self.num_nulls, input:size(2)):squeeze()
        self.null_output = self.null_table:forward(self.null_input)
    end

    -- for the non-null tokens, do regular pos forward
    self.pos_mask = input:gt(self.skip_value) -- or just 1-(null_mask)
    self.num_pos = self.pos_mask:sum()
    if(self.num_pos > 0) then
        self.pos_input = input:maskedSelect(self.pos_mask:repeatTensor(1,input:size(2))):reshape(self.num_pos, input:size(2))
        self.pos_output = self.pos_net:forward(self.pos_input)
    end

    -- combine the two outputs
    self.output = self.pos_output and self.pos_output.new(input:size(1), self.pos_output:size(2)) or self.null_output.new(input:size(1), self.null_output:size(2))
    if(self.num_nulls > 0) then
        self.output:maskedCopy(self.null_mask:repeatTensor(1,self.output:size(2)), self.null_output)
    end
    if(self.num_pos > 0) then
        self.output:maskedCopy(self.pos_mask:repeatTensor(1,self.output:size(2)), self.pos_output)
    end
    return self.output
end


function LookupWrapper:updateGradInput(input, gradOutput)
    -- do backward for just null tokens
    if(self.num_nulls > 0) then
        local null_grad_output = gradOutput.new(self.null_output:size()):fill(0)
        null_grad_output:copy(gradOutput:maskedSelect(self.null_mask:repeatTensor(1, gradOutput:size(2))))
        self.null_table:backward(self.null_input, null_grad_output)
    end

    -- do backward for just pos tokens
    if(self.num_pos > 0) then
        local pos_grad_output = gradOutput.new(self.pos_output:size()):fill(0)
        pos_grad_output:copy(gradOutput:maskedSelect(self.pos_mask:repeatTensor(1, gradOutput:size(2))))
        self.pos_net:backward(self.pos_input, pos_grad_output)
    end

    --    local pos_grad_input = self.pos_net.gradInput
    --
    --    print({pos_grad_output})
    --    print({pos_grad_input})
    --
    --    -- combine grad inputs
    --    self.gradInput = pos_grad_input.new(input[1]:size(1), pos_grad_input:size(2))
    --    self.gradInput:maskedCopy(self.null_mask:repeatTensor(1, pos_grad_input:size(2)), null_grad_input)
    --    self.gradInput:maskedCopy(self.pos_mask:repeatTensor(1, pos_grad_input:size(2)), pos_grad_input)
    --
    --    return self.gradInput
end

