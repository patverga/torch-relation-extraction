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
cmd:option('-loadModel', '', 'load a pretrained model from this file')
cmd:option('-saveModel', '', 'file to save the trained model to')
cmd:option('-loadColEncoder', '', 'load a pretrained relation/column encoder from this file')
cmd:option('-loadRowEncoder', '', 'load a pretrained entity/row encoder from this file')
cmd:option('-loadRowEmbeddings', '', 'file containing serialized torch embeddings')
cmd:option('-loadColEmbeddings', '', 'file containing serialized torch embeddings')

-- model / data sizes
cmd:option('-colEncoder', 'lookup-table', 'type of encoder to use for relations')
cmd:option('-rowEncoder', 'lookup-table', 'type of encoder to use for entities')
cmd:option('-tieEncoders', false, 'tie the row and column encoders to use the same parameters')
cmd:option('-rowDim', 100, 'embedding dimension for rows')
cmd:option('-colDim', 100, 'embedding dimension for columns')
cmd:option('-tokenDim', 100, 'embedding dimension for input tokens in compositional models')
cmd:option('-batchSize', 128, 'minibatch size')
cmd:option('-kbWeight', 1, 'weight kb relations differently than text relations')
cmd:option('-k', 3, 'k for topK model')
cmd:option('-sharedVocab', false, 'use col vocab size for rows')
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
cmd:option('-poolLayer', 'last', 'pool the outputs of each token from the lstm or convolution. valid choices are Mean and Max')
cmd:option('-relationPool', '', 'Pool relations using this - valid choices are Mean and Max')
cmd:option('-mlp', false, "add an mlp after the column and 'row' embeddings in the column only models")
cmd:option('-nonLinearLayer', '', 'apply non-linearity before pool layer')


-- conv net specific
cmd:option('-convWidth', 3, 'convolution width')

-- simple we model specific
cmd:option('-mean', false, 'use mean to combine word embeddings instead of `x')

-- rnn specific
cmd:option('-bi', '', 'Use a bidirectional lstm and combine forawrd backward in this way (add, concat, no-reverse-concat)')
cmd:option('-rnnCell', false, 'Use a regular rnn cell instead of lstm')
cmd:option('-noWordUpdate', false, 'Dont update word embeddings')

-- optimization
cmd:option('-criterion', 'bpr', 'criterion to use [bpr, hinge]')
cmd:option('-learningRate', 0.0001, 'init learning rate')
cmd:option('-decay', 0, 'learning rate decay')
cmd:option('-epsilon', 1e-8, 'epsilon parameter for adam optimization')
cmd:option('-beta1', 0.9, 'beta1 parameter for adam optimization')
cmd:option('-beta2', 0.999, 'beta2 parameter for adam optimization')
cmd:option('-momentum', 0, 'momentum value for optimization')
cmd:option('-colNorm', 0, 'if > 0, enforce max norm constraints on column embeddings')
cmd:option('-rowNorm', 0, 'if > 0, enforce max norm constraints on row embeddings')
cmd:option('-l2Reg', 0, 'l2 regularization weight')
cmd:option('-clipGrads', 1, 'clip gradients so l2 norm <= clipGrads. If <= 0, not enforced (default)')
cmd:option('-optimMethod', 'adam', 'which optimization method to use')
cmd:option('-stopEarly', false, 'stop training early if evaluation F1 goes down')
cmd:option('-margin', 1, 'size of margin if using hinge loss')
cmd:option('-p', 2, 'use the Lp distance')
cmd:option('-numEpochs', 10, 'number of epochs to train for')
cmd:option('-freezeRow', 0, 'freeze the ep embeddings for this many epochs')
cmd:option('-freezeCol', 0, 'freeze the rel embeddings for this many epochs')

-- evaluation
cmd:option('-test', '', 'torch format test file list')
cmd:option('-fb15kDir', '', 'torch format fb15k file dir')
cmd:option('-resultDir', '', 'tac eval output dir')
cmd:option('-vocab', '', 'vocab file to use for tac eval')
cmd:option('-tacYear', '', 'tac evaluation year')
cmd:option('-evalArgs', '', 'other args for tac eval')
cmd:option('-evaluateFrequency', 1, 'evaluate after every ith epoch')


function CmdArgs:parse(cmd_args)
    local params = cmd:parse(cmd_args)
    -- print the params in sorted order
    local param_array = {}
    for arg, val in pairs(params) do table.insert(param_array, arg .. ' : ' .. tostring(val)) end
    table.sort(param_array)
    for _, arg_val in ipairs(param_array) do print(arg_val) end
    return params
end