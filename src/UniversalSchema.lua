--
-- User: pat
-- Date: 8/26/15
--

package.path = package.path .. ";src/?.lua"

require 'CmdArgs'

local params = CmdArgs:parse(arg)
-- use relation vectors instead of word embeddings
params.testing = true
torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local train_data = torch.load(params.train)

local rel_size = train_data.num_rels
local rel_dim = params.relDim > 0 and params.relDim or params.embeddingDim
local rel_table = nn.LookupTable(rel_size, rel_dim)
--rel_table.weight = rel_table.weight:normal(0, 1):mul(1 / params.embeddingDim)
-- initialize in range [-.1, .1]
rel_table.weight = torch.rand(rel_size, rel_dim):add(-.5):mul(0.1)

if params.loadRelEmbeddings ~= '' then
    rel_table.weight = (torch.load(params.loadRelEmbeddings))
end


local model
if params.entityModel then
    require 'UniversalSchemaEntityEncoder'
    model = UniversalSchemaEntityEncoder(params, rel_table, rel_table, true)
else
    require 'UniversalSchemaEncoder'
    model = UniversalSchemaEncoder(params, rel_table, rel_table, true)
end

print(model.net)
model:train()
if params.saveModel ~= '' then  model:save_model(params.numEpochs) end




