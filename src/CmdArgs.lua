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
cmd:option('-saveModel', false, 'save the trained model to resultDir')
cmd:option('-noCloneSave', false, 'save as is without cloning and cpu-ing to save memory')
cmd:option('-resultDir', '', 'output dir for logs and saved models')
cmd:option('-loadColEncoder', '', 'load a pretrained relation/column encoder from this file')
cmd:option('-loadRowEncoder', '', 'load a pretrained entity/row encoder from this file')
cmd:option('-loadRowEmbeddings', '', 'file containing serialized torch embeddings')
cmd:option('-loadColEmbeddings', '', 'file containing serialized torch embeddings')
cmd:option('-colsOnly', false, 'Load cols as rows')
cmd:option('-typeSampleFile', '', 'file containing containing type sample entity pair mappings')
cmd:option('-padIdx', 2, 'index of pad token')

-- model / data sizes
cmd:option('-colEncoder', 'lookup-table', 'type of encoder to use for relations')
cmd:option('-rowEncoder', 'lookup-table', 'type of encoder to use for entities')
cmd:option('-tieEncoders', false, 'tie the row and column encoders to use the same parameters')
cmd:option('-tieColViews', false, 'tie the inout/attention and output column encoders')
cmd:option('-rowDim', 25, 'embedding dimension for rows')
cmd:option('-colDim', 25, 'embedding dimension for columns')
cmd:option('-tokenDim', 50, 'embedding dimension for input tokens in compositional models')
cmd:option('-batchSize', 1024, 'minibatch size')
cmd:option('-textWeight', 1, 'weight text relations differently than kb relations')
cmd:option('-maxKbIndex', 237, 'weight text relations differently than kb relations')
cmd:option('-k', 3, 'k for topK model')
cmd:option('-sharedVocab', false, 'use col vocab size for rows')
cmd:option('-maxSeq', 100, 'maximum sequence length')
cmd:option('-shuffle', true, 'shuffle data, read in sequence length order otherwise')
cmd:option('-negSamples', 1, 'number of negative samples to use for models that support that (most dont)')

-- uschema specific
cmd:option('-modelType', 'entity-pair', 'which type of model to use. (entity-pair, entity, joint)')
cmd:option('-compositional', false, 'add compositional layers to merge entities to entity pair')

-- nn specific
cmd:option('-dropout', 0.0, 'dropout rate after embedding layer')
cmd:option('-layerDropout', 0.0, 'dropout rate between layers')
cmd:option('-hiddenDropout', 0.0, 'dropout rate for final representation')
cmd:option('-wordDropout', 0.0, 'word dropout rate')
cmd:option('-patternDropout', 0, 'maximum number of patterns for pooling models. If greater number of patterns, apply random dropout')
cmd:option('-layers', 1, 'number of rnn layers to use')
cmd:option('-poolLayer', 'last', 'pool the outputs of each token from the lstm or convolution. valid choices are Mean and Max')
cmd:option('-relationPool', '', 'Pool relations using this - valid choices are Mean and Max')
cmd:option('-lookupWrapper', false, 'Wrap encoder')
cmd:option('-aggregationType', '', "Type of pool classifier to use")
cmd:option('-nonLinearLayer', '', 'apply non-linearity before pool layer')
cmd:option('-distanceType', 'dot', 'distance metric for certain networks [dot, cosine]')

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
cmd:option('-learningRate', 0.001, 'init learning rate')
cmd:option('-lrDecay', 0, 'learning rate decay')
cmd:option('-initializeRange', .1, 'initialize lookup tables weights to be within the range [-value,value]')
cmd:option('-epsilon', 1e-8, 'epsilon parameter for adam optimization')
cmd:option('-beta1', 0.9, 'beta1 parameter for adam optimization')
cmd:option('-beta2', 0.999, 'beta2 parameter for adam optimization')
cmd:option('-momentum', 0, 'momentum value for optimization')
cmd:option('-colNorm', 0, 'if > 0, enforce max norm constraints on column embeddings')
cmd:option('-rowNorm', 0, 'if > 0, enforce max norm constraints on row embeddings')
cmd:option('-colClamp', false, 'clamp values on column vectors')
cmd:option('-rowClamp', false, 'clamp values of row vectors')
cmd:option('-l2Reg', 0, 'l2 regularization weight')
cmd:option('-clipGrads', 0, 'clip gradients so l2 norm <= clipGrads. If <= 0, not enforced (default)')
cmd:option('-optimMethod', 'adam', 'which optimization method to use')
cmd:option('-margin', 1, 'size of margin if using hinge loss')
cmd:option('-p', 2, 'use the Lp distance')
cmd:option('-numEpochs', -1, 'number of epochs to train for, if -1 train until map or mrr goes down')
cmd:option('-freezeRow', 0, 'freeze the ep embeddings for this many epochs')
cmd:option('-freezeCol', 0, 'freeze the rel embeddings for this many epochs')

-- evaluation
cmd:option('-test', '', 'torch format test file list')
cmd:option('-accuracyTest', '', 'torch format test file list')
cmd:option('-fb15kDir', '', 'torch format fb15k file dir')
cmd:option('-evalLog', '', 'print evaluation stats from fb15k to this log')
cmd:option('-evalCutoff', 0, 'quit evaluation after this many samples - 0 = all')
cmd:option('-filterSingleRelations', false, 'filter out single relation entity pairs when doing fb15k eval')
cmd:option('-maxDecreaseEpochs', 5, 'when num epochs = -1, stop training when evaluation doesn\'t increase for this many epochs.')
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