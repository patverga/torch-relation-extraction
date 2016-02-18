local EncoderPool, parent = torch.class('nn.EncoderPool', 'nn.Container')


function EncoderPool:__init(mapper,reducer)
    parent.__init(self)

    self.mapper = mapper
    self.reducer = reducer

    self.modules = {}
    table.insert(self.modules,mapper)
    table.insert(self.modules,reducer)

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
    self.mapped = self.mapper:updateOutput(self.reshapedInput)
    self.sizes3 = self.mapped:size()

    self.sizes2 = self.sizes2 or torch.LongStorage(self.mapped:dim() + 1)
    self.sizes2[1] = minibatchSize
    self.sizes2[2] = numPerExample

    for i = 2,self.mapped:dim() do
        self.sizes2[i+1] = self.mapped:size(i)
    end

    self.mappedAndReshaped = self.mapped:view(self.sizes2)
    self.output = self.reducer:updateOutput(self.mappedAndReshaped)
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


function EncoderPool:genericBackward(operator,input, gradOutput)
    operator(self.reducer,self.mappedAndReshaped,gradOutput)
    local reducerGrad = self.reducer.gradInput

    if reducerGrad:isContiguous() then
        self.reshapedReducerGrad = reducerGrad:view(self.sizes3)
    else
        self.reshapedReducerGrad = self.reshapedReducerGrad or reducerGrad:clone()
        self.reshapedReducerGrad:resizeAs(reducerGrad):copy(reducerGrad):resize(self.sizes3)
    end

    operator(self.mapper,self.reshapedInput,self.reshapedReducerGrad)
    local mapperGrad = self.mapper.gradInput

    self.gradInput = (mapperGrad:dim() > 0) and mapperGrad:view(self.inputSize) or nil --some modules return nil from backwards, such as the lookup table
    return self.gradInput
end