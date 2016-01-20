--
-- User: pat
-- Date: 1/8/16
--

package.path = package.path .. ";src/?.lua"

require 'rnn'
require 'torch'
require 'nn-modules/EncoderDecoder'


opt = {}
opt.learningRate = 0.1
opt.hiddenSz = 2
opt.vocabSz = 5
opt.inputSeqLen = 3 -- length of the encoded sequence


-- Some example data
 encInSeq = torch.Tensor({{1,2,3},{3,2,1}})
 decInSeq = torch.Tensor({{1,2,3,4},{4,3,2,1}})
decOutSeq = torch.Tensor({{2,3,4,1},{1,2,4,3}})

decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)


--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function forwardConnect(encLSTM, decLSTM)
 decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.inputSeqLen])
 decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.inputSeqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function backwardConnect(encLSTM, decLSTM)
 encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
 encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end


-- Encoder
enc = nn.Sequential()
enc:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
enc:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
encLSTM = nn.FastLSTM(opt.hiddenSz, opt.hiddenSz)
enc:add(nn.Sequencer(encLSTM))
enc:add(nn.SelectTable(-1))

-- Decoder
dec = nn.Sequential()
dec:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
dec:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
decLSTM = nn.FastLSTM(opt.hiddenSz, opt.hiddenSz)
dec:add(nn.Sequencer(decLSTM))
dec:add(nn.Sequencer(nn.Linear(opt.hiddenSz, opt.vocabSz)))
dec:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

encParams, encGradParams = enc:getParameters()
decParams, decGradParams = dec:getParameters()

enc:zeroGradParameters()
dec:zeroGradParameters()


-- Forward pass
--local encOut = enc:forward(encInSeq)
--forwardConnect(encLSTM, decLSTM)
--local decOut = dec:forward(decInSeq)
--local Edec = criterion:forward(decOut, decOutSeq)
---- Backward pass
--local gEdec = criterion:backward(decOut, decOutSeq)
--print(Edec,gEdec[1])
--
--dec:backward(decInSeq, gEdec)
--backwardConnect(encLSTM, decLSTM)
--local zeroTensor = torch.Tensor(2):zero()
--enc:backward(encInSeq, zeroTensor)


 encoder_decoder = nn.EncoderDecoder(enc, encLSTM, dec, decLSTM)

for i = 1, 3 do
    encoder_decoder:zeroGradParameters()
     out = encoder_decoder:forward({encInSeq, decInSeq})
    criterion:forward(out[2], decOutSeq)
     Edec = criterion:forward(out[2], decOutSeq)
     gEdec = criterion:backward(out[2], decOutSeq)
    print(Edec, gEdec[1])

    encoder_decoder:backward(decInSeq, gEdec)

    --print(enc:forward(encInSeq))
    print(encoder_decoder.encoderNet:forward(encInSeq))
    encoder_decoder:updateParameters(.1)
end

