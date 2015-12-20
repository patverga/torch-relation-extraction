--
-- User: pat
-- Date: 10/2/15
--
package.path = package.path .. ";src/?.lua"
require 'CmdArgs'
require 'EncoderFactory'
require 'rnn'

local params = CmdArgs:parse(arg)
torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

--local train_data = torch.load(params.train)
--
--local function build_lstm_uschema_model()
--    local inputSize = params.wordDim > 0 and params.wordDim or (params.relDim > 0 and params.relDim or params.embeddingDim)
--    local outputSize = params.relDim > 0 and params.relDim or params.embeddingDim
--
--    local rel_size = train_data.num_tokens
--    local word_table
--    -- never update word embeddings, these should be preloaded
--    if params.noWordUpdate then
--        require 'nn-modules/NoUpdateLookupTable'
--        word_table = nn.NoUpdateLookupTable(rel_size, inputSize)
--    else
--        word_table = nn.LookupTable(rel_size, inputSize)
--    end
--
--    -- initialize in range [-.1, .1]
--    word_table.weight = torch.rand(rel_size, inputSize):add(-.5):mul(0.1)
--    if params.loadRelEmbeddings ~= '' then
--        word_table.weight = (torch.load(params.loadRelEmbeddings))
--    end
--
--    local text_encoder = nn.Sequential()
--    text_encoder:add(word_table)
--    -- instead of updating the word embeddings, learn a linear transform for them
--    if params.noWordUpdate then
--        text_encoder:add(nn.TemporalConvolution(inputSize, inputSize, 1))
--    end
--    if params.dropout > 0.0 then
--        text_encoder:add(nn.Dropout(params.dropout))
--    end
--    text_encoder:add(nn.SplitTable(2)) -- tensor to table of tensors
--
--    -- recurrent layer
--    local lstm = nn.Sequential()
--    for i = 1, params.layers do
--        local recurrent_cell
--        if params.rnnCell then
--            recurrent_cell = nn.Recurrent(outputSize, nn.Linear(inputSize, outputSize), nn.Linear(outputSize, outputSize), nn.Sigmoid(), 9999)
--        else
--            recurrent_cell = nn.FastLSTM(i == 1 and inputSize or outputSize, outputSize)
--            --        recurrent_cell.forgetGate:get(1):get(1).bias:fill(1)
--        end
--        if params.bi then
--            --        lstm:add(nn.BiSequencer(recurrent_cell, recurrent_cell:clone()))
--            require 'nn-modules/NoUnReverseBiSequencer.lua'
--            lstm:add(nn.NoUnReverseBiSequencer(recurrent_cell, recurrent_cell:clone()))
--        else
--            lstm:add(nn.Sequencer(recurrent_cell))
--        end
--    end
--    text_encoder:add(lstm)
--
--    if params.attention then
--        require 'nn-modules/ViewTable'
--        require 'nn-modules/ReplicateAs'
--        require 'nn-modules/SelectLast'
--        require 'nn-modules/VariableLengthJoinTable'
--        require 'nn-modules/VariableLengthConcatTable'
--
--        local mixture_dim = outputSize
--        local M = nn.Sequential()
--        local term_1 = nn.Sequential()
--        term_1:add(nn.TemporalConvolution(outputSize, mixture_dim, 1))
--
--        local term_2_linear = nn.Sequential()
--        term_2_linear:add(nn.SelectLast(2))
--        term_2_linear:add(nn.Linear(mixture_dim, mixture_dim))
--
--        local term_2_concat = nn.VariableLengthConcatTable()
--        term_2_concat:add(term_2_linear)
--        term_2_concat:add(nn.Identity())
--
--        local term_2 = nn.Sequential()
--        term_2:add(term_2_concat)
--        term_2:add(nn.ReplicateAs(2, 2))
--
--        local M_concat = nn.VariableLengthConcatTable():add(term_1):add(term_2)
--        M:add(M_concat):add(nn.CAddTable())
--
--        local Y = nn.Identity()
--        local alpha = nn.Sequential():add(M):add(nn.TemporalConvolution(mixture_dim,1,1)):add(nn.Select(3,1)):add(nn.SoftMax()):add(nn.Replicate(1,2))
--        local concat_table = nn.ConcatTable():add(alpha):add(Y)
--
--        local attention = nn.Sequential()
--        attention:add(concat_table)
--        attention:add(nn.MM())
--        --    attention:add(nn.MixtureTable())
--
--        text_encoder:add(nn.ViewTable(-1, 1, outputSize))
--        text_encoder:add(nn.VariableLengthJoinTable(2))
--        text_encoder:add(attention)
--        text_encoder:add(nn.View(-1, mixture_dim))
--
--    elseif params.poolLayer ~= '' then
--        assert(params.poolLayer == 'Mean' or params.poolLayer == 'Max',
--            'valid options for poolLayer are Mean and Max')
--        require 'nn-modules/ViewTable'
--        text_encoder:add(nn.ViewTable(-1, 1, outputSize))
--        text_encoder:add(nn.JoinTable(2))
--        text_encoder:add(nn[params.poolLayer](2))
--    else
--        text_encoder:add(nn.SelectTable(-1))
--    end
--    return text_encoder
--end

--local function build_uschema_model()
--    local rel_size = train_data.num_tokens
----    local rel_dim = params.relDim > 0 and params.relDim or params.embeddingDim
--    local rel_dim = params.embeddingDim
--    local kb_rel_table = nn.LookupTable(rel_size, rel_dim)
--    --rel_table.weight = rel_table.weight:normal(0, 1):mul(1 / params.embeddingDim)
--    -- initialize in range [-.1, .1]
--    kb_rel_table.weight = torch.rand(rel_size, rel_dim):add(-.5):mul(0.1)
--
----    if params.loadRelEmbeddings ~= '' then
----        kb_rel_table.weight = (torch.load(params.loadRelEmbeddings))
----    end
--    return kb_rel_table
--end

-- initialize the two models
params.encoder = 'lstm'
local text_encoder, kb_rel_table = EncoderFactory:build_encoder(params)
params.encoder = 'lookup-table'
local kb_rel_table, _ = EncoderFactory:build_encoder(params)
--local kb_rel_table = text_encoder:get(1):clone()

require 'UniversalSchemaJointEncoder'
local model = UniversalSchemaJointEncoder(params, kb_rel_table, text_encoder)
model:train()
