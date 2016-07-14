local EncoderFactory = torch.class('EncoderFactory')


function EncoderFactory:build_lookup_table(params, load_embeddings, vocab_size, dim)
    local pre_trained_embeddings = load_embeddings ~= '' and torch.load(load_embeddings)
    vocab_size = pre_trained_embeddings and math.max(vocab_size, pre_trained_embeddings:size(1)) or vocab_size
    local lookup_table
    -- never update word embeddings, these should be preloaded
    if params.noWordUpdate then
        lookup_table = nn.NoUpdateLookupTable(vocab_size, dim)
    else
        lookup_table = nn.LookupTableMaskPad(vocab_size, dim, params.padIdx)
    end

    if pre_trained_embeddings and lookup_table.weight:size(2) == pre_trained_embeddings:size(2) then
        lookup_table.weight = pre_trained_embeddings
    -- preload the first k dimensionsa nd randomly initialize the rest
    elseif pre_trained_embeddings and lookup_table.weight:size(2) > pre_trained_embeddings:size(2) then
        print ('Warning : Pretrained embeddings smaller than specified dimension - randomly initializing missing parameters')
        local missing_parameters = torch.rand(vocab_size, dim-pre_trained_embeddings:size(2)):mul(params.initializeRange):add(-params.initializeRange/2):float()
        lookup_table.weight = pre_trained_embeddings:cat(missing_parameters, 2)
    else
        lookup_table.weight = torch.rand(vocab_size, dim):mul(params.initializeRange):add(-params.initializeRange/2)
    end
    -- preload first k embeddings and randomly initialize the rest
    if lookup_table.weight:size(1) < vocab_size then
        print('Warning : Number of pretrained embeddings is smaller than vocab size, randomly initializing missing embeddings.')
        local missing_rows = torch.rand(vocab_size - lookup_table.weight:size(1), dim):mul(params.initializeRange):add(-params.initializeRange/2):float()
        lookup_table.weight = lookup_table.weight:cat(missing_rows, 1)
    end
    return lookup_table
end

function EncoderFactory:lstm_encoder(params, load_embeddings, vocab_size, embedding_dim)
    local input_dim = params.tokenDim > 0 and params.tokenDim or embedding_dim
    local output_dim = embedding_dim
    local lookup_table = self:build_lookup_table(params, load_embeddings, vocab_size, input_dim)

    local encoder = nn.Sequential()
    -- word dropout
    if params.wordDropout > 0 then
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
            params.rnnCell and nn.Recurrent(layer_output_size, nn.Linear(layer_input_size, layer_output_size),
            nn.Linear(layer_output_size, layer_output_size), nn.Sigmoid(), 9999) -- regular rnn
            or nn.FastLSTMNoBias(layer_input_size, layer_output_size):maskZero(2) -- lstm
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
        assert(params.poolLayer == 'Mean' or params.poolLayer == 'Max' or params.poolLayer == 'last',
            'valid options for poolLayer are Mean, Max, and last')
        if params.poolLayer == 'last' then
            encoder:add(nn.SelectTable(-1))
        else
            encoder:add(nn.ViewTable(-1, 1, output_dim))
            encoder:add(nn.JoinTable(2))
--            if params.nonLinearLayer ~= '' then encoder:add(nn[params.nonLinearLayer]()) end
            encoder:add(nn[params.poolLayer](2))
        end
    end

    return encoder, lookup_table
end


function EncoderFactory:seq_lstm_encoder(params, load_embeddings, vocab_size, embedding_dim)
    local input_dim = params.tokenDim > 0 and params.tokenDim or embedding_dim
    local output_dim = embedding_dim
    local lookup_table = self:build_lookup_table(params, load_embeddings, vocab_size, input_dim)

    local encoder = nn.Sequential()
    -- word dropout
    if params.wordDropout > 0 then
        encoder:add(nn.WordDropout(params.wordDropout, 1))
    end

    encoder:add(lookup_table)
    if params.dropout > 0.0 then encoder:add(nn.Dropout(params.dropout)) end
--    encoder:add(nn.Print('embeddings', true))
    -- recurrent layer
    local lstm = nn.Sequential()
    for i = 1, params.layers do
        local layer_output_size = (i < params.layers or not string.find(params.bi, 'concat')) and output_dim or output_dim / 2
        local layer_input_size = i == 1 and input_dim or output_dim
        local lstm_layer = nn.SeqLSTM(layer_input_size, layer_output_size)
        lstm_layer.batchfirst = true
        lstm_layer.maskzero = true
        lstm:add(lstm_layer)
        if params.layerDropout > 0.0 then lstm:add(nn.Sequencer(nn.Dropout(params.layerDropout))) end
    end
    encoder:add(lstm)
    encoder:add(nn.SplitTable(2))

    -- pool hidden units of sequence to get single vector or take last
    if params.poolLayer ~= '' then
        assert(params.poolLayer == 'Mean' or params.poolLayer == 'Max' or params.poolLayer == 'last',
            'valid options for poolLayer are Mean, Max, and last')
        if params.poolLayer == 'last' then
            encoder:add(nn.SelectTable(-1))
        else
            encoder:add(nn.ViewTable(-1, 1, output_dim))
            encoder:add(nn.JoinTable(2))
            --            if params.nonLinearLayer ~= '' then encoder:add(nn[params.nonLinearLayer]()) end
            encoder:add(nn[params.poolLayer](2))
        end
    end

    return encoder, lookup_table
end


function EncoderFactory:cnn_encoder(params, load_embeddings, vocab_size, embedding_dim)
    local input_dim = params.tokenDim > 0 and params.tokenDim or embedding_dim
    local output_dim = embedding_dim
    local lookup_table = self:build_lookup_table(params, load_embeddings, vocab_size, input_dim)

    local encoder = nn.Sequential()
    if params.wordDropout > 0 then
        encoder:add(nn.WordDropout(params.wordDropout, 1))
    end
    encoder:add(lookup_table)
    if params.dropout > 0.0 then
        encoder:add(nn.Dropout(params.dropout))
    end
    if (params.convWidth > 1) then encoder:add(nn.Padding(1, 1, -1)) end
    if (params.convWidth > 2) then encoder:add(nn.Padding(1, -1, -1)) end
    encoder:add(nn.TemporalConvolution(input_dim, output_dim, params.convWidth))
    encoder:add(nn.Tanh())
    local pool_layer = params.poolLayer ~= '' and params.poolLayer or 'Max'
    encoder:add(nn[pool_layer](2))

    return encoder, lookup_table
end

function EncoderFactory:we_avg_encoder(params, load_embeddings, vocab_size, embedding_dim)
    local dim = params.tokenDim > 0 and params.tokenDim or embedding_dim
    local lookup_table = self:build_lookup_table(params, load_embeddings, vocab_size, dim)

    local encoder = nn.Sequential()
    encoder:add(lookup_table)
    local pool_layer = params.poolLayer ~= '' and params.poolLayer or 'Mean'
    encoder:add(nn[pool_layer](2))

    return encoder, lookup_table
end

function EncoderFactory:lookup_table_split(params, load_embeddings, vocab_size, embedding_dim)
    local dim = params.tokenDim > 0 and params.tokenDim or embedding_dim
    local lookup_table = self:build_lookup_table(params, load_embeddings, vocab_size, dim)
    local encoder = nn.Sequential():add(lookup_table):add(nn.SplitTable(2, embedding_dim))
    return encoder, lookup_table
end

function EncoderFactory:lstm_joint_encoder(params)
    local text_encoder, _ = self:lstm_encoder(params)
    local kb_encoder, _ = self:lookup_table_encoder(params)
    return text_encoder, kb_encoder
end

function EncoderFactory:relation_pool_encoder(params, sub_encoder)
    local encoder = nn.Sequential()
    if params.relationPool == 'Mean' or params.relationPool == 'Max' then
        encoder:add(nn.EncoderPool(sub_encoder, nn[params.relationPool](2)))
--        if params.colEncoder == 'lookup-table' then encoder:add(nn.View(-1, params.colDim)) end
    elseif params.relationPool == 'identity' or params.relationPool == 'Identity' then
        encoder:add(nn.EncoderPool(sub_encoder, nn.Identity()))
    elseif params.relationPool == 'convolution' then
        encoder:add(nn.EncoderPool(sub_encoder, nn.TemporalConvolution(params.colDim, params.colDim, 1)))
    elseif params.relationPool == 'conv-max' then
        encoder:add(nn.EncoderPool(sub_encoder, nn.Identity()))
            :add(nn.TemporalConvolution(params.colDim, params.colDim, 1)):add(nn.Max(2))
    else
        print('valid options for relationPool are Mean and Max')
        os.exit()
    end
    return encoder
end

function EncoderFactory:build_encoder(params, encoder_type, load_embeddings, vocab_size, embedding_dim)
    local encoder, table

    -- lstm encoder
    if encoder_type == 'lstm' then
        encoder, table = self:lstm_encoder(params, load_embeddings, vocab_size, embedding_dim)

    -- faster lstm encoder that computes full sequence at once
    elseif encoder_type == 'seq-lstm' then
        encoder, table = self:seq_lstm_encoder(params, load_embeddings, vocab_size, embedding_dim)

            -- conv net
    elseif encoder_type == 'cnn' then
        encoder, table = self:cnn_encoder(params, load_embeddings, vocab_size, embedding_dim)

        -- simple token averaging
    elseif encoder_type == 'we-avg' then
        encoder, table = self:we_avg_encoder(params, load_embeddings, vocab_size, embedding_dim)

        -- lstm for text, lookup-table for kb relations
    elseif encoder_type == 'lstm-joint' then
        encoder, table = self:lstm_joint_encoder(params, load_embeddings, vocab_size, embedding_dim)

        -- lookup table for transe and uschema entity
    elseif encoder_type == 'lookup-table-split' then
        encoder, table = self:lookup_table_split(params, load_embeddings, vocab_size, embedding_dim)

        -- lookup table (vector per relation)
    elseif encoder_type == 'lookup-table' then
        local lookup_table = self:build_lookup_table(params, load_embeddings, vocab_size, embedding_dim)
        encoder = nn.Sequential():add(lookup_table); table = lookup_table
    else
        print('Must supply option to encoder. ' ..
                'Valid options are: lstm, cnn, we-avg, lstm-joint, lstm-relation-pool, lookup-table, and lookup-table-split')
        os.exit()
    end

    if params.hiddenDropout > 0 then encoder:add(nn.Dropout(params.hiddenDropout)) end
    return encoder, table
end
