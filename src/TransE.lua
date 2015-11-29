--
-- User: pat
-- Date: 8/26/15
--

require 'TransEEncoder'
require 'CmdArgs'
require 'torch'

local params = CmdArgs:parse(arg)
-- use relation vectors instead of word embeddings
params.testing = true

torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local train_data = torch.load(params.train)

local rel_table = nn.LookupTable(train_data.num_rels, params.embeddingDim)
-- initialize in range [-.1, .1]
rel_table.weight = torch.rand(train_data.num_rels, params.embeddingDim):add(-.5):mul(0.1)

if params.loadRelEmbeddings ~= '' then
    rel_table.weight = (torch.load(params.loadRelEmbeddings))
end


local model = TransEEncoder(params, rel_table, rel_table, true)
print(model.net)
model:train()
model:evaluate()


