--
-- User: pat
-- Date: 8/26/15
--

package.path = package.path .. ";src/?.lua"

require 'CmdArgs'
require 'EncoderFactory'
require 'rnn'
require 'UniversalSchemaEncoder'
require 'UniversalSchemaRelationPooling'
require 'UniversalSchemaEntityEncoder'
require 'UniversalSchemaJointEncoder'
require 'TransEEncoder'

local params = CmdArgs:parse(arg)
-- use relation vectors instead of word embeddings
torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local train_data = torch.load(params.train)

local function get_encoder(encoder_type, vocab_size, dim, load_encoder, load_embeddings)
    local encoder, table
    if load_encoder ~= '' then -- load encoder from saved model
        local loaded_model = torch.load(load_encoder)
        encoder, table = loaded_model.col_encoder, loaded_model.col_table
    else
        encoder, table = EncoderFactory:build_encoder(params, encoder_type, load_embeddings, vocab_size, dim)
    end
    return encoder, table
end


local col_encoder, col_table, row_encoder, row_table
if params.tieEncoders then -- use the same encoder for columns and rows
    -- handle old and new data formats
    local vocab_size = train_data.num_rels and (params.colEncoder == 'lookup-table' and math.max(train_data.num_rels, train_data.num_eps) or train_data.num_tokens)
--    or (params.colEncoder == 'lookup-table' and math.max(train_data.num_cols, train_data.num_rows) or math.max(train_data.num_col_tokens, train_data.num_row_tokens))
    or (params.colEncoder == 'lookup-table' and train_data.num_cols or math.max(train_data.num_col_tokens, train_data.num_row_tokens))

    col_encoder, col_table = get_encoder(params.colEncoder, vocab_size, params.colDim, params.loadColEncoder, params.loadColEmbeddings)
    row_encoder, row_table = col_encoder:clone(), col_table:clone()
    col_table:share(row_table, 'weight', 'bias', 'gradWeight', 'gradBias')
    col_encoder:share(row_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')
else
    -- create column encoder
    local col_vocab_size = train_data.num_eps and (params.colEncoder == 'lookup-table' and train_data.num_rels or train_data.num_tokens)
            or (params.colEncoder == 'lookup-table' and train_data.num_cols or train_data.num_col_tokens)
    col_encoder, col_table = get_encoder(params.colEncoder, col_vocab_size, params.colDim, params.loadColEncoder, params.loadColEmbeddings)

    -- create row encoder
    local row_vocab_size = train_data.num_eps and (params.rowEncoder == 'lookup-table' and train_data.num_eps or train_data.num_tokens)
            or (params.rowEncoder == 'lookup-table' and train_data.num_rows or train_data.num_row_tokens)
    row_encoder, row_table = get_encoder(params.rowEncoder, row_vocab_size, params.rowDim, params.loadRowEncoder, params.loadRowEmbeddings)
end

-- pool all relations for given ep and udpate at once
-- requires processing data using bin/process/IntFile2PoolRelationsTorch.lua
if params.relationPool and params.relationPool ~= '' then
    col_encoder = EncoderFactory:relation_pool_encoder(params, col_encoder)
end



local model
-- learn vectors for each entity rather than entity pair
if params.modelType == 'entity' then
    model = UniversalSchemaEntityEncoder(params, row_table, row_encoder, col_table, col_encoder, true)

-- use a lookup table for kb relations and encoder for text patterns (entity pair vectors)
elseif params.modelType == 'joint' then
    model = UniversalSchemaJointEncoder(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'transE' then -- standard uschema with entity pair vectors
    model = TransEEncoder(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'max' then
    model = UniversalSchemaMax(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'mean' then
    model = UniversalSchemaMean(params, row_table, row_encoder, col_table, col_encoder, false)

    -- TODO figure out how to do this with autograd
--elseif params.modelType == 'topK' then
--    model = UniversalSchemaTopK(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'attention-dot' then
    model = UniversalSchemaAttentionDot(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'attention-matrix' then
    model = UniversalSchemaAttentionMatrix(params, row_table, row_encoder, col_table, col_encoder, false)

--elseif params.relationPool ~= '' then
--    model = UniversalSchemaRelationPool(params, row_table, row_encoder, col_table, col_encoder, false)

else -- standard uschema with entity pair vectors
    model = UniversalSchemaEncoder(params, row_table, row_encoder, col_table, col_encoder, false)
end

print(model.net)
if params.numEpochs > 0 then
    model:train()
    if params.saveModel ~= '' then  model:save_model(params.numEpochs) end
else model:evaluate() end
