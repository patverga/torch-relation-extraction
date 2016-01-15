--
-- User: pat
-- Date: 1/8/16
--

package.path = package.path .. ";src/?.lua"

require 'rnn'
require 'torch'
require 'EncoderDecoder'


opt = {}
opt.learningRate = 0.1
opt.hiddenSz = 2
opt.vocabSz = 5
opt.inputSeqLen = 3 -- length of the encoded sequence


-- Some example data
 encInSeq, decInSeq, decOutSeq = torch.Tensor({{1,2,3},{3,2,1}}), torch.Tensor({{1,2,3,4},{4,3,2,1}}), torch.Tensor({{2,3,4,1},{1,2,4,3}})
decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)

-- Encoder
 enc = nn.Sequential()
enc:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
enc:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
 encLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
enc:add(nn.Sequencer(encLSTM))
enc:add(nn.SelectTable(-1))

-- Decoder
 dec = nn.Sequential()
dec:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
dec:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
 decLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
dec:add(nn.Sequencer(decLSTM))
dec:add(nn.Sequencer(nn.Linear(opt.hiddenSz, opt.vocabSz)))
dec:add(nn.Sequencer(nn.LogSoftMax()))

 criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

 encParams, encGradParams = enc:getParameters()
 decParams, decGradParams = dec:getParameters()

enc:zeroGradParameters()
dec:zeroGradParameters()


 encoder_decoder = nn.EncoderDecoder(enc, encLSTM, dec, decLSTM)

 out = encoder_decoder:forward({encInSeq, decInSeq})
print (out)
criterion:forward(out[2], decOutSeq)
 Edec = criterion:forward(out[2], decOutSeq)
 gEdec = criterion:backward(out[2], decOutSeq)
print(encoder_decoder:backward(decInSeq, gEdec))
