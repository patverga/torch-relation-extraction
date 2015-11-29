--
-- User: pat
-- Date: 8/26/15
--
package.path = package.path .. ";src/?.lua"

require 'CmdArgs'

local params = CmdArgs:parse(arg)
torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local train_data = torch.load(params.train)
local inputSize = params.wordDim > 0 and params.wordDim or (params.relDim > 0 and params.relDim or params.embeddingDim)

local rel_size = train_data.num_tokens
local rel_table
-- never update word embeddings, these should be preloaded
if params.noWordUpdate then
    require 'nn-modules/NoUpdateLookupTable'
    rel_table = nn.NoUpdateLookupTable(rel_size, inputSize)
else
    rel_table = nn.LookupTable(rel_size, inputSize)
end

-- initialize in range [-.1, .1]
rel_table.weight = torch.rand(rel_size, inputSize):add(-.5):mul(0.1)
if params.loadRelEmbeddings ~= '' then
    rel_table.weight = (torch.load(params.loadRelEmbeddings))
end


local encoder = nn.Sequential()
encoder:add(rel_table)
local pool_layer = params.poolLayer ~= '' and params.poolLayer or 'Mean'
encoder:add(nn[pool_layer](2))


local model
if params.entityModel then
    require 'UniversalSchemaEntityEncoder'
    model = UniversalSchemaEntityEncoder(params, rel_table, encoder)
else
    require 'UniversalSchemaEncoder'
    model = UniversalSchemaEncoder(params, rel_table, encoder)
end
print(model.net)
model:train()
if params.saveModel ~= '' then  model:save_model(params.numEpochs) end




