local ReplicateAs, parent = torch.class('nn.ReplicateAs','nn.Module')

function ReplicateAs:__init(dim, ndim)
    parent.__init(self)
    self.dim = dim or 1
    self.ndim = ndim
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    assert(self.dim > 0, "Can only replicate across positive integer dimensions.")
end

function ReplicateAs:updateOutput(input)
--    print(input)
    self.nfeatures = input[2]:size(2)
    local input = input[1]
--    assert(
--        self.dim <= input:dim()+1,
--        "Not enough input dimensions to replicate along dimension " ..
--                tostring(self.dim) .. ".")
    local batchOffset = self.ndim and input:dim() > self.ndim and 1 or 0
    local rdim = self.dim + batchOffset
    local sz = torch.LongStorage(input:dim()+1)
    sz[rdim] = self.nfeatures
    for i = 1,input:dim() do
        local offset = 0
        if i >= rdim then
            offset = 1
        end
        sz[i+offset] = input:size(i)
    end
    local st = torch.LongStorage(input:dim()+1)
    st[rdim] = 0
    for i = 1,input:dim() do
        local offset = 0
        if i >= rdim then
            offset = 1
        end
        st[i+offset] = input:stride(i)
    end
    self.output = input.new(input:storage(),input:storageOffset(),sz,st)
    self.output = torch.reshape(self.output, self.output:size(1), self.output:size(3)*self.output:size(2))
    return self.output
end

function ReplicateAs:updateGradInput(input, gradOutput)
--    self.zeroGrad = self.zeroGrad or input[2]:clone():fill(0)
--    self.dim = input[2]:size(self.dimtocopyfrom)
    local inputForSize = input[2]
    local input = input[1]
    self.gradInput[1]:resizeAs(input):zero()
    self.gradInput[2]:resizeAs(inputForSize):zero()

    local batchOffset = self.ndim and input:dim() > self.ndim and 1 or 0
    local rdim = self.dim + batchOffset
    local sz = torch.LongStorage(input:dim()+1)
    sz[rdim] = 1
    for i = 1,input:dim() do
        local offset = 0
        if i >= rdim then
            offset = 1
        end
        sz[i+offset] = input:size(i)
    end

--print (self.gradInput, {input,gradOutput}, rdim, sz)

    local gradInput = self.gradInput[1]:view(sz)
    gradInput:sum(gradOutput, rdim)
    return self.gradInput
end

