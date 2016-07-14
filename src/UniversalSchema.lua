
package.path = package.path .. ";src/?.lua;src/nn-modules/?.lua"

require 'CmdArgs'
require 'EncoderFactory'
require 'rnn'
require 'UniversalSchemaEncoder'
require 'UniversalSchemaEntityEncoder'
require 'TransEEncoder'
require 'UniversalSchemaNegativeSample'
require 'UniversalOvals'
require 'nn_modules_init'

local params = CmdArgs:parse(arg)
torch.manualSeed(0)

print('Using ' .. (params.gpuid >= 0 and 'GPU' or 'CPU'))
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

if params.modelType == 'oval' then params.rowDim = params.rowDim * 2; params.colDim = params.colDim * 2 end -- embedding is mean:cat(variance)

local col_encoder, col_table, row_encoder, row_table
if params.loadModel == '' then
    local train_data = torch.load(params.train)

    local function get_encoder(encoder_type, vocab_size, dim, load_encoder, load_embeddings, row)
        local encoder, table
        if load_encoder ~= '' then -- load encoder from saved model
            local loaded_model = torch.load(load_encoder)
            if row then encoder, table = loaded_model.row_encoder, loaded_model.row_table
            else encoder, table = loaded_model.col_encoder, loaded_model.col_table end
            -- assuming 1 lookup table / encoder
            table = table or encoder:findModules('nn.LookupTableMaskPad')[1]
        else
            encoder, table = EncoderFactory:build_encoder(params, encoder_type, load_embeddings, vocab_size, dim)
        end
        return encoder, table
    end


    --[[ Define column and row encoders ]]--
    if params.tieEncoders then -- use the same encoder for columns and rows
        -- handle old and new data formats
        local vocab_size = train_data.num_rels and (params.colEncoder == 'lookup-table' and math.max(train_data.num_rels, train_data.num_eps) or train_data.num_tokens)
                --    or (params.colEncoder == 'lookup-table' and math.max(train_data.num_cols, train_data.num_rows) or math.max(train_data.num_col_tokens, train_data.num_row_tokens))
                or (params.colEncoder == 'lookup-table' and train_data.num_cols or math.max(train_data.num_col_tokens, train_data.num_row_tokens))

        col_encoder, col_table = get_encoder(params.colEncoder, vocab_size, params.colDim, params.loadColEncoder, params.loadColEmbeddings)
        row_encoder, row_table = col_encoder:clone('weight', 'bias', 'gradWeight', 'gradBias'), (col_table and col_table:clone('weight', 'bias', 'gradWeight', 'gradBias'))
    else
        -- create column encoder
        local col_vocab_size = train_data.num_eps and (params.colEncoder == 'lookup-table' and train_data.num_rels or train_data.num_tokens)
                or (params.colEncoder == 'lookup-table' and train_data.num_cols or train_data.num_col_tokens)
        col_encoder, col_table = get_encoder(params.colEncoder, col_vocab_size, params.colDim, params.loadColEncoder, params.loadColEmbeddings)

        -- create row encoder
        local row_vocab_size = params.sharedVocab and col_vocab_size or
                (train_data.num_eps and (params.rowEncoder == 'lookup-table' and train_data.num_eps or train_data.num_tokens) -- 4col format
                        or (params.rowEncoder == 'lookup-table' and train_data.num_rows or train_data.num_row_tokens)) -- 3col format
        row_encoder, row_table = get_encoder(params.rowEncoder, row_vocab_size, params.rowDim, params.loadRowEncoder, params.loadRowEmbeddings, true and not params.colsOnly)
    end

    if params.relationPool ~= '' and #col_encoder:findModules('nn.EncoderPool') == 0 then col_encoder = nn.EncoderPool(col_encoder:clearState()) end
    if params.lookupWrapper then
        local kb_table =  nn.LookupTable(params.maxKbIndex, params.colDim)
        kb_table.weight = torch.rand(params.maxKbIndex, params.colDim):mul(params.initializeRange):add(-params.initializeRange/2)
        col_encoder = nn.LookupWrapper(col_encoder,kb_table, params.maxKbIndex, 1)
    end
end


--[[ Define model type ]]--
local model
-- learn vectors for each entity rather than entity pair
if params.modelType == 'entity' then
    model = UniversalSchemaEntityEncoder(params, row_table, row_encoder, col_table, col_encoder, true)

elseif params.modelType == 'transE' then -- standard uschema with entity pair vectors
    model = TransEEncoder(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'entity-pair' then -- standard uschema with entity pair vectors
    model = UniversalSchemaEncoder(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'neg-sample' then
    model = UniversalSchemaNegativeSample(params, row_table, row_encoder, col_table, col_encoder, false)

elseif params.modelType == 'weight-kb' then
    model = WeightedKBTraining(params, row_table, row_encoder, col_table, col_encoder, false)


else
    print('Must supply option to modelType. Valid options are: '
            .. 'entity-pair, entity, transE, max, mean, attention-dot, and attention-matrix')
    os.exit()
end


print(model.net)
if params.numEpochs ~= 0 then model:train()
else model:evaluate(params.numEpochs) end
