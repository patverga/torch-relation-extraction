--
-- User: pat
-- Date: 7/27/15
--

require 'torch'

local CmdArgs = torch.class('CmdArgs')

local cmd = torch.CmdLine()
-- use cuda?
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')

-- data file locations
cmd:option('-train', 'data/train-mtx.torch', 'torch format train file list')
cmd:option('-test', '', 'torch format test file list')
cmd:option('-loadModel', '', 'load a pretrained model from this file')
cmd:option('-saveModel', '', 'file to save the trained model to')
cmd:option('-loadEncoder', '', 'load a pretrained encoder from this file')
cmd:option('-loadEpEmbeddings', '', 'file containing serialized torch embeddings')
cmd:option('-loadRelEmbeddings', '', 'file containing serialized torch embeddings')
cmd:option('-saveEpEmbeddings', '', 'write entity pair embeddings to text file')
cmd:option('-saveRelEmbeddings', '', 'write relation embeddings to text file')

-- model / data sizes
cmd:option('-encoder', '', 'embedding dimension')
cmd:option('-embeddingDim', 100, 'embedding dimension')
cmd:option('-relDim', 0, 'embedding dimension')
cmd:option('-wordDim', 0, 'embedding dimension')
cmd:option('-batchSize', 128, 'minibatch size')
cmd:option('-margin', 1, 'size of margin')
cmd:option('-p', 2, 'use the Lp distance')
cmd:option('-kbWeight', 1, 'weight kb relations differently than text relations')
cmd:option('-l2Reg', 0, 'l2 regularization weight')
cmd:option('-maxSeq', 100, 'maximum sequence length')
cmd:option('-shuffle', true, 'shuffle data, read in sequence length order otherwise')

-- uschema specific
cmd:option('-modelType', 'entity-pair', 'which type of model to use. (entity-pair, entity, joint)')
cmd:option('-compositional', false, 'add compositional layers to merge entities to entity pair')

-- nn specific
cmd:option('-dropout', 0.0, 'dropout rate after embedding layer')
cmd:option('-layerDropout', 0.0, 'dropout rate between layers')
cmd:option('-wordDropout', 0.0, 'word dropout rate')
cmd:option('-layers', 1, 'number of rnn layers to use')
cmd:option('-poolLayer', '', 'pool the outputs of each token from the lstm or convolution. valid choices are Mean and Max')


-- conv net specific
cmd:option('-convWidth', 3, 'convolution width')

-- simple we model specific
cmd:option('-mean', false, 'use mean to combine word embeddings instead of `x')

-- rnn specific
cmd:option('-bi', '', 'Use a bidirectional lstm and combine forawrd backward in this way (add, concat, no-reverse-concat)')
cmd:option('-rnnCell', false, 'Use a regular rnn cell instead of lstm')
cmd:option('-attention', false, 'Use an attention model')
cmd:option('-noWordUpdate', false, 'Dont update word embeddings')
cmd:option('-poolRelations', false, 'pool all relations for given ep and udpate at once, requires processing data using bin/process/IntFile2PoolRelationsTorch.lua')

-- optimization
cmd:option('-learningRate', 0.0001, 'init learning rate')
cmd:option('-decay', 0, 'learning rate decay')
cmd:option('-epsilon', 1e-8, 'epsilon parameter for adam optimization')
cmd:option('-beta1', 0.9, 'beta1 parameter for adam optimization')
cmd:option('-beta2', 0.999, 'beta2 parameter for adam optimization')
cmd:option('-momentum', 0, 'momentum value for optimization')
cmd:option('-regularize', false, 'use max norm constraints')
cmd:option('-clipGrads', 1, 'clip gradients so l2 norm <= clipGrads. If <= 0, not enforced (default)')
cmd:option('-optimMethod', 'adam', 'which optimization method to use')
cmd:option('-stopEarly', false, 'stop training early if evaluation F1 goes down')
cmd:option('-numEpochs', 10, 'number of epochs to train for')
cmd:option('-freezeEp', 0, 'freeze the ep embeddings for this many epochs')
cmd:option('-freezeRel', 0, 'freeze the rel embeddings for this many epochs')

-- evaluation
--cmd:option('-k', 0, 'export the top k relations for each of the test relations')
cmd:option('-resultDir', '', 'tac eval output dir')
cmd:option('-vocab', '', 'vocab file to use for tac eval')
cmd:option('-tacYear', '', 'tac evaluation year')
cmd:option('-evalArgs', '', 'other args for tac eval')
cmd:option('-evaluateFrequency', 1, 'evaluate after every ith epoch')

-- other
--cmd:option('-profile', false, 'if true, run pepperfish profiler')
cmd:option('-testing', false, 'run in testing mode')

function CmdArgs:parse(arg)
    local params = cmd:parse(arg)
    print(params)
    return params
end