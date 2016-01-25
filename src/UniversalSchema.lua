--
-- User: pat
-- Date: 8/26/15
--

package.path = package.path .. ";src/?.lua"

require 'CmdArgs'
require 'EncoderFactory'
require 'rnn'

local params = CmdArgs:parse(arg)
-- use relation vectors instead of word embeddings
torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local train_data = torch.load(params.train)

-- create column encoder
local rel_encoder, rel_table
if params.loadRelEncoder ~= '' then -- load encoder from saved model
    local loaded_model = torch.load(params.loadRelEncoder)
    rel_encoder, rel_table = loaded_model.rel_encoder, loaded_model.rel_table
else
    local rel_vocab_size = params.encoder == 'lookup-table' and train_data.num_rels or train_data.num_tokens
    rel_encoder, rel_table = EncoderFactory:build_encoder(params, params.encoder, rel_vocab_size, params.relDim)
    if params.loadRelEmbeddings ~= '' then rel_table.weight = (torch.load(params.loadRelEmbeddings)) end
end

-- create row encoder
local ent_encoder, ent_table
if params.loadEntEncoder ~= '' then -- load encoder from saved model
    local loaded_model = torch.load(params.loadEntEncoder)
    ent_encoder, ent_table = loaded_model.ent_encoder, loaded_model.ent_table
else
    local ent_vocab_size = params.entEncoder == 'lookup-table' and train_data.num_eps or train_data.num_eps --num_tokens
    ent_encoder, ent_table = EncoderFactory:build_encoder(params, params.entEncoder, ent_vocab_size, params.embeddingDim)
    if params.loadEpEmbeddings ~= '' then ent_table.weight = (torch.load(params.loadEpEmbeddings)) end
end


local model
-- learn vectors for each entity rather than entity pair
if params.modelType == 'entity' then
    require 'UniversalSchemaEntityEncoder'
    model = UniversalSchemaEntityEncoder(params, ent_table, ent_encoder, rel_table, rel_encoder, true)

-- use a lookup table for kb relations and encoder for text patterns (entity pair vectors)
elseif params.modelType == 'joint' then
    require 'UniversalSchemaJointEncoder'
    model = UniversalSchemaJointEncoder(params, ent_table, ent_encoder, rel_table, rel_encoder, false)

-- standard uschema with entity pair vectors
else
    require 'UniversalSchemaEncoder'
    model = UniversalSchemaEncoder(params, ent_table, ent_encoder, rel_table, rel_encoder, false)
end

print(model.net)
model:train()
if params.saveModel ~= '' then  model:save_model(params.numEpochs) end
