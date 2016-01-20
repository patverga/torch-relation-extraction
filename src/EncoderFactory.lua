
local EncoderFactory = torch.class('EncoderFactory')

function EncoderFactory:lstm_encoder(params)
    local train_data = torch.load(params.train)

    local input_dim = params.wordDim > 0 and params.wordDim or (params.relDim > 0 and params.relDim or params.embeddingDim)
    local output_dim = params.relDim > 0 and params.relDim or params.embeddingDim

    local vocab_size = train_data.num_tokens
    local lookup_table = self:build_lookup_table(params, vocab_size, input_dim)

    local encoder = nn.Sequential()
    -- word dropout
    if params.wordDropout > 0 then
        require 'nn-modules/WordDropout'
        encoder:add(nn.WordDropout(params.wordDropout, 1))
    end

    encoder:add(lookup_table)
    if params.dropout > 0.0 then encoder:add(nn.Dropout(params.dropout)) end
    encoder:add(nn.SplitTable(2)) -- tensor to table of tensors

    -- recurrent layer
    local lstm = nn.Sequential()
    for i = 1, params.layers do
        local layer_output_size = (i < params.layers or not string.find(params.bi, 'concat')) and output_dim or output_dim / 2
        local layer_input_size = i == 1 and input_dim or output_dim
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

    -- pool hidden units of sequence to get single vector or take last
    if params.poolLayer ~= '' then
        assert(params.poolLayer == 'Mean' or params.poolLayer == 'Max',
            'valid options for poolLayer are Mean and Max')
        require 'nn-modules/ViewTable'
        encoder:add(nn.ViewTable(-1, 1, output_dim))
        encoder:add(nn.JoinTable(2))
        if params.nonLinearLayer ~= '' then encoder:add(nn[params.nonLinearLayer]()) end
        encoder:add(nn[params.poolLayer](2))
    else
        encoder:add(nn.SelectTable(-1))
    end

    return encoder, lookup_table
end

function EncoderFactory:relation_pool_encoder(params, sub_encoder, lookup_table)
    require 'nn-modules/EncoderPool'
    assert(params.relationPool == 'Mean' or params.relationPool == 'Max',
        'valid options for poolLayer are Mean and Max')
    local encoder = nn.EncoderPool(sub_encoder, nn[params.relationPool](2))
    return encoder, lookup_table
end

function EncoderFactory:cnn_encoder(params)
    local train_data = torch.load(params.train)

    local input_dim = params.wordDim > 0 and params.wordDim or (params.relDim > 0 and params.relDim or params.embeddingDim)
    local output_dim = params.relDim > 0 and params.relDim or params.embeddingDim

    local vocab_size = train_data.num_tokens
    local lookup_table = self:build_lookup_table(params, vocab_size, input_dim)

    local encoder = nn.Sequential()
    if params.wordDropout > 0 then
        require 'nn-modules/WordDropout'
        encoder:add(nn.WordDropout(params.wordDropout, 1))
    end
    encoder:add(lookup_table)
    if params.dropout > 0.0 then
        encoder:add(nn.Dropout(params.dropout))
    end
    if (params.convWidth > 1) then encoder:add(nn.Padding(1,1,-1)) end
    if (params.convWidth > 2) then encoder:add(nn.Padding(1,-1,-1)) end
    encoder:add(nn.TemporalConvolution(input_dim, output_dim, params.convWidth))
    encoder:add(nn.Tanh())
    local pool_layer = params.poolLayer ~= '' and params.poolLayer or 'Max'
    encoder:add(nn[pool_layer](2))

    return encoder, lookup_table
end

function EncoderFactory:we_avg_encoder(params)
    local train_data = torch.load(params.train)

    local vocab_size = train_data.num_tokens
    local dim = params.wordDim > 0 and params.wordDim or (params.relDim > 0 and params.relDim or params.embeddingDim)
    local lookup_table = self:build_lookup_table(params, vocab_size, dim)

    local encoder = nn.Sequential()
    encoder:add(lookup_table)
    local pool_layer = params.poolLayer ~= '' and params.poolLayer or 'Mean'
    encoder:add(nn[pool_layer](2))

    return encoder, lookup_table
end

function EncoderFactory:build_lookup_table(params, vocab_size, dim)
    local lookup_table
    -- never update word embeddings, these should be preloaded
    if params.noWordUpdate then
        require 'nn-modules/NoUpdateLookupTable'
        lookup_table = nn.NoUpdateLookupTable(vocab_size, dim)
    else
        lookup_table = nn.LookupTable(vocab_size, dim)
    end
    -- initialize in range [-.1, .1]
    lookup_table.weight = torch.rand(vocab_size, dim):add(-.1):mul(0.1)

    if params.loadRelEmbeddings ~= '' then
        lookup_table.weight = (torch.load(params.loadRelEmbeddings))
    end
    return lookup_table
end


function EncoderFactory:lookup_table_encoder(params)
    local train_data = torch.load(params.train)
    local vocab_size = train_data.num_rels
    local dim = params.relDim > 0 and params.relDim or params.embeddingDim
    local rel_table = self:build_lookup_table(params, vocab_size, dim)
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
--
--function EncoderFactory:lstm_lstm_attention_encoder(params)
--    local attention_lstm, attention_rel_table = self:lstm_encoder(params)
--    local lstm, rel_table = self:lstm_encoder(params)
--
--    local encoder = nn.Sequntial()
--
--end


function EncoderFactory:build_encoder(params)
    local encoder_type = params.encoder

    local encoder, table
    -- load encoder from saved model
    if params.loadEncoder ~= '' then
        local loaded_model = torch.load(params.loadEncoder)
        encoder, table = loaded_model.encoder, loaded_model.rel_table

    -- lstm encoder
    elseif encoder_type == 'lstm' then
        encoder, table = self:lstm_encoder(params)

    -- conv net
    elseif encoder_type == 'cnn' then
        encoder, table = self:cnn_encoder(params)

    -- simple token averaging
    elseif encoder_type == 'we-avg' then
        encoder, table = self:we_avg_encoder(params)

    -- lstm for text, lookup-table for kb relations
    elseif encoder_type == 'lstm-joint' then
        encoder, table = self:lstm_joint_encoder(params)

    -- lookup table (vector per relation)
    elseif encoder_type == 'lookup-table' then
        params.relations = true
        encoder, table = self:lookup_table_encoder(params)
    else
        print('Must supply option to encoder. ' ..
        'Valid options are: lstm, cnn, we-avg, lstm-joint, lstm-relation-pool, and lookup-table')
        os.exit()
    end

    -- pool all relations for given ep and udpate at once
    -- requires processing data using bin/process/IntFile2PoolRelationsTorch.lua
    if params.relationPool ~= "" then
        encoder, table = self:relation_pool_encoder(params, encoder, table)
    end

    return encoder, table
end
