local EncoderPool, parent = torch.class('nn.EncoderPool', 'nn.Container')


function EncoderPool:__init(encoder, pooler)
    parent.__init(self)

    self.encoder = encoder
    self.pooler = pooler

    self.modules = {}
    table.insert(self.modules,encoder)
    table.insert(self.modules,pooler)

end

function EncoderPool:updateOutput(input)
    --first, reshape the data by pulling the second dimension into the first

    self.inputSize = input:size()
    local numPerExample = self.inputSize[2]
    local minibatchSize = self.inputSize[1]
    self.sizes = self.sizes or torch.LongStorage(self.inputSize:size() -1)
    self.sizes[1] = minibatchSize*numPerExample
    for i = 2,self.sizes:size() do
        self.sizes[i] = self.inputSize[i+1]
    end

    self.reshapedInput = input:view(self.sizes)
    self.mapped = self.encoder:updateOutput(self.reshapedInput)
    self.sizes3 = self.mapped:size()

    self.sizes2 = self.sizes2 or torch.LongStorage(self.mapped:dim() + 1)
    self.sizes2[1] = minibatchSize
    self.sizes2[2] = numPerExample

    for i = 2,self.mapped:dim() do
        self.sizes2[i+1] = self.mapped:size(i)
    end

    self.mappedAndReshaped = self.mapped:view(self.sizes2)
    self.output = self.pooler:updateOutput(self.mappedAndReshaped)
    return self.output

end

function EncoderPool:backward(input,gradOutput)
    local function operator(module,input,gradOutput) return module:backward(input,gradOutput) end
    return self:genericBackward(operator,input,gradOutput)
end


function EncoderPool:updateGradInput(input,gradOutput)
    local function operator(module,input,gradOutput) return module:updateGradInput(input,gradOutput) end
    return self:genericBackward(operator,input,gradOutput)
end

function EncoderPool:accUpdateGradParameters(input,gradOutput,lr)
    local function operator(module,input,gradOutput) return module:accUpdateGradParameters(input,gradOutput,lr) end
    return self:genericBackward(operator,input,gradOutput)
end

function EncoderPool:accGradParameters(input,gradOutput,lr)
    local function operator(module,input,gradOutput) return module:accGradParameters(input,gradOutput,lr) end
    return self:genericBackward(operator,input,gradOutput)
end


function EncoderPool:genericBackward(operator, input, gradOutput)
    local db = self.pooler:forward(self.mappedAndReshaped)
    self.pooler:backward(self.mappedAndReshaped,db:clone():fill(1.0))
    operator(self.pooler,self.mappedAndReshaped,gradOutput)
    local poolerGrad = self.pooler.gradInput
    local reshapedPoolerGrad = poolerGrad:view(self.sizes3)


    operator(self.encoder,self.reshapedInput,reshapedPoolerGrad)
    local encoderGrad = self.encoder.gradInput

    self.gradInput = (encoderGrad:dim() > 0) and encoderGrad:view(self.inputSize) or nil --some modules return nil from backwards, such as the lookup table
    return self.gradInput
end