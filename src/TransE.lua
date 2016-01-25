--
-- User: pat
-- Date: 8/26/15
--
package.path = package.path .. ";src/?.lua"

require 'TransEEncoder'
require 'EncoderFactory'
require 'CmdArgs'
require 'torch'

local params = CmdArgs:parse(arg)
-- use relation vectors instead of word embeddings
params.relations = true

torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local encoder, col_table = EncoderFactory:build_encoder(params)


local model = TransEEncoder(params, col_table, encoder)
print(model.net)
model:train()
model:evaluate()


