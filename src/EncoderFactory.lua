
local EncoderFactory = torch.class('EncoderFactory')

function EncoderFactory:lstm_encoder(params)
    local train_data = torch.load(params.train)

    local encoder
    local rel_table
    if params.loadEncoder ~= '' then
        local loaded_model = torch.load(params.loadEncoder)
        encoder = loaded_model.encoder
        rel_table = loaded_model.rel_table:clone()
    else
        local inputSize = params.wordDim > 0 and params.wordDim or (params.relDim > 0 and params.relDim or params.embeddingDim)
        local outputSize = params.relDim > 0 and params.relDim or params.embeddingDim

        local rel_size = train_data.num_tokens
        -- never update word embeddings, these should be preloaded
        if params.noWordUpdate then
            require 'nn-modules/NoUpdateLookupTable'
            rel_table = nn.NoUpdateLookupTable(rel_size, inputSize):add(nn.TemporalConvolution(inputSize, inputSize, 1))
        else
            rel_table = nn.LookupTable(rel_size, inputSize)
        end

        -- initialize in range [-.1, .1]
        rel_table.weight = torch.rand(rel_size, inputSize):add(-.5):mul(0.1)
        if params.loadRelEmbeddings ~= '' then
            rel_table.weight = (torch.load(params.loadRelEmbeddings))
        end

        encoder = nn.Sequential()
        -- word dropout
        if params.wordDropout > 0 then
            require 'nn-modules/WordDropout'
            encoder:add(nn.WordDropout(params.wordDropout, 1))
        end

        encoder:add(rel_table)
        if params.dropout > 0.0 then encoder:add(nn.Dropout(params.dropout)) end
        encoder:add(nn.SplitTable(2)) -- tensor to table of tensors

        -- recurrent layer
        local lstm = nn.Sequential()
        for i = 1, params.layers do
            local layer_output_size = (i < params.layers or not string.find(params.bi, 'concat')) and outputSize or outputSize / 2
            local layer_input_size = i == 1 and inputSize or outputSize
            local recurrent_cell =
                -- regular rnn
            params.rnnCell and nn.Recurrent(layer_output_size, nn.Linear(layer_input_size, layer_output_size),
                nn.Linear(layer_output_size, layer_output_size), nn.Sigmoid(), 9999)
                    -- lstm
                    or nn.FastLSTM(layer_input_size, layer_output_size)
            if params.bi == "add" then
                lstm:add(nn.BiSequencer(recurrent_cell, recurrent_cell:clone(), nn.CAddTable()))
            elseif params.bi == "linear" then
                lstm:add(nn.Sequential():add(nn.BiSequencer(recurrent_cell, recurrent_cell:clone())):add
                (nn.Sequencer(nn.Linear(layer_output_size*2, layer_output_size))))
                --        elseif params.bi == "concat" then
                --            lstm:add(nn.BiSequencer(recurrent_cell, recurrent_cell:clone()))
                --        elseif params.bi == "no-reverse-concat" then
                --            require 'nn-modules/NoUnReverseBiSequencer'
                --            lstm:add(nn.NoUnReverseBiSequencer(recurrent_cell, recurrent_cell:clone()))
            else
                lstm:add(nn.Sequencer(recurrent_cell))
            end
            if params.layerDropout > 0.0 then lstm:add(nn.Sequencer(nn.Dropout(params.layerDropout))) end
        end
        encoder:add(lstm)

        if params.poolLayer ~= '' then
            assert(params.poolLayer == 'Mean' or params.poolLayer == 'Max',
                'valid options for poolLayer are Mean and Max')
            require 'nn-modules/ViewTable'
            encoder:add(nn.ViewTable(-1, 1, outputSize))
            encoder:add(nn.JoinTable(2))
            encoder:add(nn[params.poolLayer](2))
        else
            encoder:add(nn.SelectTable(-1))
        end
    end
    return encoder, rel_table
end

function EncoderFactory:lstm_relation_pool_encoder(params)
    local lstm, rel_table = self:lstm_encoder(params)
    require 'nn-modules/EncoderPool'
    local encoder = nn.EncoderPool(lstm, nn.Max(2))
    return encoder, rel_table
end

function EncoderFactory:cnn_encoder(params)
    local train_data = torch.load(params.train)

    local inputSize = params.wordDim > 0 and params.wordDim or (params.relDim > 0 and params.relDim or params.embeddingDim)
    local outputSize = params.relDim > 0 and params.relDim or params.embeddingDim

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
    if params.wordDropout > 0 then
        require 'nn-modules/WordDropout'
        encoder:add(nn.WordDropout(params.wordDropout, 1))
    end
    encoder:add(rel_table)
    if params.dropout > 0.0 then
        encoder:add(nn.Dropout(params.dropout))
    end
    if (params.convWidth > 1) then encoder:add(nn.Padding(1,1,-1)) end
    if (params.convWidth > 2) then encoder:add(nn.Padding(1,-1,-1)) end
    encoder:add(nn.TemporalConvolution(inputSize, outputSize, params.convWidth))
    encoder:add(nn.Tanh())
    local pool_layer = params.poolLayer ~= '' and params.poolLayer or 'Max'
    encoder:add(nn[pool_layer](2))

    return encoder, rel_table
end

function EncoderFactory:we_avg_encoder(params)
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

    return encoder, rel_table
end

function EncoderFactory:lookup_table_encoder(params)
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

    return rel_table, rel_table
end

function EncoderFactory:lstm_joint_encoder(params)
    local text_encoder, _ = self:lstm_encoder(params)
    local kb_rel_table, _ = self:lookup_table_encoder(params)
    return text_encoder, kb_rel_table

end



-- TODO set this up
--function EncoderFactory:attention_encoder(params)
--    if params.attention then
--        require 'nn-modules/ViewTable'
--        require 'nn-modules/ReplicateAs'
--        require 'nn-modules/SelectLast'
--        require 'nn-modules/VariableLengthJoinTable'
--        require 'nn-modules/VariableLengthConcatTable'
--
--        local mixture_dim = outputSize
--        local M = nn.Sequential()
--        local term_1 = nn.Sequential()
--        term_1:add(nn.TemporalConvolution(outputSize, mixture_dim, 1))
--
--        local term_2_linear = nn.Sequential()
--        term_2_linear:add(nn.SelectLast(2))
--        term_2_linear:add(nn.Linear(mixture_dim, mixture_dim))
--
--        local term_2_concat = nn.VariableLengthConcatTable()
--        term_2_concat:add(term_2_linear)
--        term_2_concat:add(nn.Identity())
--
--        local term_2 = nn.Sequential()
--        term_2:add(term_2_concat)
--        term_2:add(nn.ReplicateAs(2, 2))
--
--        local M_concat = nn.VariableLengthConcatTable():add(term_1):add(term_2)
--        M:add(M_concat):add(nn.CAddTable())
--
--        local Y = nn.Identity()
--        local alpha = nn.Sequential():add(M):add(nn.TemporalConvolution(mixture_dim,1,1)):add(nn.Select(3,1)):add(nn.SoftMax()):add(nn.Replicate(1,2))
--        local concat_table = nn.ConcatTable():add(alpha):add(Y)
--
--        local attention = nn.Sequential()
--        attention:add(concat_table)
--        attention:add(nn.MM())
--        --    attention:add(nn.MixtureTable())
--
--        encoder:add(nn.ViewTable(-1, 1, outputSize))
--        encoder:add(nn.VariableLengthJoinTable(2))
--        encoder:add(attention)
--        encoder:add(nn.View(-1, mixture_dim))
--end


function EncoderFactory:build_encoder(params)
    local encoder_type = params.encoder

    -- lstm encoder
    if encoder_type == 'lstm' then
        return self:lstm_encoder(params)

    -- conv net
    elseif encoder_type == 'cnn' then
        return self:cnn_encoder(params)

    -- simple token averaging
    elseif encoder_type == 'we-avg' then
        return self:we_avg_encoder(params)

    -- lstm for text, lookup-table for kb relations
    elseif encoder_type == 'lstm-joint' then
        return self:lstm_joint_encoder(params)

    -- pool all relations for given ep and udpate at once,
    -- requires processing data using bin/process/IntFile2PoolRelationsTorch.lua
    elseif encoder_type == 'lstm-relation-pool' then
        return self:lstm_relation_pool_encoder(params)

    -- lookup table (vector per relation)
    elseif encoder_type == 'lookup-table' then
        params.relations = true
        return self:lookup_table_encoder(params)

    else
        print('Must supply option to encoder. ' ..
        'Valid options are: lstm, cnn, we-avg, lstm-joint, lstm-relation-pool, and lookup-table')
        os.exit()
    end
end
