--
-- User: pat
-- Date: 9/17/15
--

--[[
Takes a tac candidate file, tab seperated vocab idx file, and a trained uschema encoder model
and exports a scored candidtate file to outfile
]]--

package.path = package.path .. ";src/?.lua;src/nn-modules/?.lua;src/eval/?.lua;src/classifier/?.lua;"

require 'torch'
require 'rnn'
require 'nn_modules_init'
require 'RelationPoolFactory'
require 'ScoringFunctions'
require 'TacEvalCmdArgs'

grad = require 'autograd'
grad.optimize(true) -- global


local params = TacEvalCmdArgs:parse(arg)

if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

-- load model
local model = torch.load(params.model)
local kb_encoder = model.kb_col_table and model.kb_col_table
        or (model.row_encoder and model.row_encoder
        or (model.col_encoder and model.col_encoder
        or model.encoder))
local text_encoder = model.text_encoder and model.text_encoder
        or (model.col_encoder and model.col_encoder
        or model.encoder)
local net = model.net
net:evaluate(); kb_encoder:evaluate(); text_encoder:evaluate()


---- main
local scorer
if params.scoringType == 'cosine' then
    -- directly compare column representations
    kb_encoder = text_encoder:clone()
    scorer = CosineSentenceScorer(params, net, kb_encoder, text_encoder)
elseif params.scoringType == 'classifier' then
    scorer = SentenceClassifier(params, net, kb_encoder, text_encoder)
elseif params.scoringType == 'pool-classifier' then
    scorer = PoolSentenceClassifier(params, net, kb_encoder, text_encoder)
elseif params.scoringType == 'network' then
    scorer = NetworkScorer(params, net, kb_encoder, text_encoder)
else
    print('Must supply valid scoringType : cosine, network, classifier, pool-classifier')
    os.exit()
end

scorer:run()
