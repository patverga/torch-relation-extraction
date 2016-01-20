local EncoderDecoder, parent = torch.class('nn.EncoderDecoder', 'nn.Container')


function EncoderDecoder:__init(encoderNet, encoderLSTM, decoderNet, decoderLSTM)
    parent.__init(self)

    self.encoderNet = encoderNet
    self.encoderLSTM = encoderLSTM
    self.decoderNet = decoderNet
    self.decoderLSTM = decoderLSTM
    self.modules = {encoderNet, decoderNet}
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function EncoderDecoder:forwardConnect(encLSTM, decLSTM)
    decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[#encLSTM.outputs])
    decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[#encLSTM.cells])
end


--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function EncoderDecoder:backwardConnect(encLSTM, decLSTM)
    encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
    encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end

function EncoderDecoder:updateOutput(input)
    -- Forward pass
    local encOut = self.encoderNet:forward(input[1])
    self:forwardConnect(self.encoderLSTM, self.decoderLSTM)
    local decOut = self.decoderNet:forward(input[2])
    self.output = {encOut, decOut}
    return self.output
end

function EncoderDecoder:backward(input,gradOutput)
    -- Backward pass
    self.decoderNet:backward(input[2], gradOutput)
    self:backwardConnect(self.encoderLSTM, self.decoderLSTM)
    local zeroTensor = torch.Tensor(2):zero()
    self.encoderNet:backward(input[1], zeroTensor)
end

function EncoderDecoder:updateGradInput(input, gradOutput)
    self.encoderNet:updateGradInput(input, gradOutput)
    self.decoderNet:updateGradInput(input, gradOutput)
end

function EncoderDecoder:accUpdateGradParameters(input,gradOutput,lr)
    self.encoderNet:accUpdateGradParameters(input,gradOutput,lr)
    self.decoderNet:accUpdateGradParameters(input,gradOutput,lr)
end

function EncoderDecoder:accGradParameters(input,gradOutput,lr)
    self.encoderNet:accGradParameters(input,gradOutput,lr)
    self.decoderNet:accGradParameters(input,gradOutput,lr)
end

