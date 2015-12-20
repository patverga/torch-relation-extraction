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


local encoder, rel_table = EncoderFactory:build_encoder(params)

local model
-- learn vectors for each entity rather than entity pair
if params.modelType == 'entity' then
    require 'UniversalSchemaEntityEncoder'
    model = UniversalSchemaEntityEncoder(params, rel_table, encoder)

-- use a lookup table for kb relations and encoder for text patterns (entity pair vectors)
elseif params.modelType == 'joint' then
    require 'UniversalSchemaJointEncoder'
    model = UniversalSchemaJointEncoder(params, rel_table, encoder)

-- standard uschema with entity pair vectors
else
    require 'UniversalSchemaEncoder'
    model = UniversalSchemaEncoder(params, rel_table, encoder)
end

print(model.net)
model:train()
if params.saveModel ~= '' then  model:save_model(params.numEpochs) end




