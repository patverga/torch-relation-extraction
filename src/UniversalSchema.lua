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
local rel_vocab_size = params.encoder == 'lookup-table' and train_data.num_rels or train_data.num_tokens
local rel_encoder, rel_table = EncoderFactory:build_encoder(params, params.encoder, rel_vocab_size)
local ent_vocab_size = params.entEncoder == 'lookup-table' and train_data.num_eps or train_data.num_eps --num_tokens
local ent_encoder, ent_table = EncoderFactory:build_encoder(params, params.entEncoder, ent_vocab_size)

-- learn vectors for each entity rather than entity pair
local model
if params.modelType == 'entity' then
    require 'UniversalSchemaEntityEncoder'
    model = UniversalSchemaEntityEncoder(params, rel_table, rel_encoder)

-- use a lookup table for kb relations and encoder for text patterns (entity pair vectors)
elseif params.modelType == 'joint' then
    require 'UniversalSchemaJointEncoder'
    model = UniversalSchemaJointEncoder(params, rel_table, rel_encoder)

-- standard uschema with entity pair vectors
else
    require 'UniversalSchemaEncoder'
    model = UniversalSchemaEncoder(params, ent_table, ent_encoder, rel_table, rel_encoder)
end

--print (rel_encoder)
--print (rel_encoder(torch.Tensor(1,1):fill(1)))
--print (ent_encoder)
--print (ent_encoder(torch.Tensor(1,1):fill(1)))

print(model.net)
model:train()
if params.saveModel ~= '' then  model:save_model(params.numEpochs) end
