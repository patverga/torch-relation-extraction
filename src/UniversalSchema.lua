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
local col_encoder, col_table
if params.loadColEncoder ~= '' then -- load encoder from saved model
    local loaded_model = torch.load(params.loadColEncoder)
    col_encoder, col_table = loaded_model.col_encoder, loaded_model.col_table
else
    local rel_vocab_size = params.colEncoder == 'lookup-table' and train_data.num_rels or train_data.num_tokens
    col_encoder, col_table = EncoderFactory:build_encoder(params, params.colEncoder, rel_vocab_size, params.colDim)
    if params.loadColEmbeddings ~= '' then col_table.weight = (torch.load(params.loadColEmbeddings)) end
end

-- create row encoder
local row_encoder, row_table
if params.loadRowEncoder ~= '' then -- load encoder from saved model
    local loaded_model = torch.load(params.loadRowEncoder)
    row_encoder, row_table = loaded_model.row_encoder, loaded_model.row_table
else
    local ent_vocab_size = params.rowEncoder == 'lookup-table' and train_data.num_eps or train_data.num_eps --num_tokens
    row_encoder, row_table = EncoderFactory:build_encoder(params, params.rowEncoder, ent_vocab_size, params.rowDim)
    if params.loadRowEmbeddings ~= '' then row_table.weight = (torch.load(params.loadRowEmbeddings)) end
end


local model
-- learn vectors for each entity rather than entity pair
if params.modelType == 'entity' then
    require 'UniversalSchemaEntityEncoder'
--    local col_encoder = nn.Sequential():add(col_encoder):add(nn.View(-1, params.colDim)):add(nn.Linear(params.colDim, params.colDim *2))
    model = UniversalSchemaEntityEncoder(params, row_table, row_encoder, col_table, col_encoder, true)

-- use a lookup table for kb relations and encoder for text patterns (entity pair vectors)
elseif params.modelType == 'joint' then
    require 'UniversalSchemaJointEncoder'
    model = UniversalSchemaJointEncoder(params, row_table, row_encoder, col_table, col_encoder, false)

-- standard uschema with entity pair vectors
else
    require 'UniversalSchemaEncoder'
    model = UniversalSchemaEncoder(params, row_table, row_encoder, col_table, col_encoder, false)
end

print(model.net)
model:train()
if params.saveModel ~= '' then  model:save_model(params.numEpochs) end
