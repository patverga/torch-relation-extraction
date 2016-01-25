--
-- User: pv
-- Date: 9/21/15
--
local RelationEncoderModel = torch.class('RelationEncoderModel')


function RelationEncoderModel:__init(params, row_table, row_encoder, col_table, col_encoder, use_entities)
    self.__index = self
    self.params = params
    self.opt_config = { learningRate = self.params.learningRate, epsilon = self.params.epsilon,
        beta1 = self.params.beta1, beta2 = self.params.beta2,
        momentum = self.params.momentum, learningRateDecay = self.params.decay
    }
    self.opt_state = {}
    self.squeeze_rel = params.relations or false
    self.train_data = self:load_train_data(params.train, use_entities)

    -- cosine distance network for evaluation
    self.cosine = self:to_cuda(nn.CosineDistance())

    -- either load model from file or initialize new one
    if params.loadModel ~= '' then
        local loaded_model = torch.load(params.loadModel)
        self.net = self:to_cuda(loaded_model.net)
        col_encoder = self:to_cuda(loaded_model.col_encoder)
        row_encoder = self:to_cuda(loaded_model.row_encoder)
        row_table = self:to_cuda(loaded_model.row_table)
        col_table = self:to_cuda(loaded_model.col_table)
        self.opt_state = loaded_model.opt_state
        for key, val in pairs(loaded_model.opt_state) do if (torch.type(val) == 'torch.DoubleTensor') then self.opt_state[key] = self:to_cuda(val) end; end
    else
        self.net = self:build_network(row_encoder, col_encoder)
    end
    self.row_table = row_table
    self.col_table = col_table
    self.col_encoder = col_encoder
    self.row_encoder = row_encoder
end


--- Utils ---

function RelationEncoderModel:to_cuda(x) return self.params.gpuid >= 0 and x:cuda() or x:double() end


--- Train ---

function RelationEncoderModel:train(num_epochs)
    num_epochs = num_epochs or self.params.numEpochs
    -- optim stuff
    local parameters, gradParameters = self.net:getParameters()

    -- make sure model save dir exists
    if self.params.saveModel ~= '' then os.execute("mkdir -p " .. self.params.saveModel) end
    for epoch = 1, num_epochs
    do
        local startTime = sys.clock()
        local batches = self:gen_training_batches(self.train_data)
        local shuffle = torch.randperm(#batches)
        local epoch_error = 0
        print('Starting epoch '.. epoch .. ' of ' .. num_epochs, '\n')
        for i = 1, #batches
        do
            local batch = self.params.shuffle and batches[shuffle[i]] or batches[i]
            local example = batch.data
            local label = batch.label
            epoch_error = epoch_error + self:optim_update(self.net, self.crit,
                example, label, parameters, gradParameters, self.opt_config, self.opt_state, epoch)

            if (i % 25 == 0) then
                io.write(string.format('\r%.2f percent complete\tspeed = %.2f examples/sec\terror = %.4f',
                    100 * i / (#batches), (i * self.params.batchSize) / (sys.clock() - startTime), (epoch_error / i)))
                io.flush()
            end
        end
        print(string.format('\nEpoch error = %f', epoch_error))
        if (epoch % self.params.evaluateFrequency == 0) then
            self:evaluate()
            self:save_model(epoch-1)
        end
    end
    self:evaluate()
    self:save_model(num_epochs)
end


--- Evaluate ----

function RelationEncoderModel:evaluate()
    if self.params.test ~= '' then
        self:map(self.params.test, true)
    end
end


function RelationEncoderModel:score_test_data(data)
    local scores = {}
    local labels = {}
    if #data > 0 or data.ep == nil then
        for seq_size = 1, data.max_length do
            local sub_data = data[seq_size]
            if sub_data and sub_data.ep then
                local score, label = self:score_subdata(sub_data, scores, labels)
                table.insert(scores, score)
                table.insert(labels, label)
            end
        end
    else
        local score, label = self:score_subdata(data, scores, labels)
        table.insert(scores, score)
        table.insert(labels, label)
    end
    scores = nn.JoinTable(1)(nn.FlattenTable()(scores))
    labels = nn.JoinTable(1)(labels)
    return scores:view(scores:size(1)), labels
end

function RelationEncoderModel:map(fileStr, high_score)
    print('Calculating MAP')
    local map = 0.0
    local file_count = 0
    for file in string.gmatch(fileStr, "[^,]+") do
        local ap = self:avg_precision(file, high_score)
        print(file, ap)
        map = map + ap
        file_count = file_count + 1
    end
    map = map / math.max(1.0, file_count)
    print('MAP : ' .. map)
end

function RelationEncoderModel:avg_precision(file, high_score)
    local correct_label = 1.0
    --    local correct_label = 2.0
    local data = torch.load(file)

    -- score each of the test samples
    local scores, labels = self:score_test_data(data)
    -- sort scores and attach the original labels correctly
    local sorted_scores, sorted_idx = torch.sort(scores, 1, high_score)
    local sorted_labels = labels:index(1, sorted_idx)
    -- TODO : stupid calc, use torch
    local ap = 0.0
    local pos_n = 0
    -- iterate over all the scored data
    for rank = 1, sorted_labels:size(1) do
        -- if label is true, increment positive count and update avg p
        if (sorted_labels[rank] == correct_label) then
            ap = ((ap * pos_n) / (pos_n + 1)) + (1.0 / rank)
            pos_n = pos_n + 1
        end
    end
    return ap
end


function RelationEncoderModel:tac_eval(model_file, out_dir, eval_args)
    if self.params.vocab ~= '' and self.params.tacYear ~= '' then
        os.execute("mkdir -p " .. model_file)
        local cmd = '${TH_RELEX_ROOT}/bin/tac-evaluation/tune-thresh.sh ' .. self.params.tacYear .. ' ' ..
                model_file..'-model' .. ' ' .. self.params.vocab .. ' ' .. self.params.gpuid ..' ' ..
                self.params.maxSeq .. ' ' .. out_dir .. ' "' .. eval_args:gsub(',',' ') ..
                '" >& ' .. model_file .. '/tac-eval.log &'
        print(cmd)
        os.execute(cmd)
    end
end

--- IO ----

function RelationEncoderModel:load_sub_data(sub_data, entities)
    if entities then
        if self.params.rowEncoder == 'lookup-table' then
            sub_data.e1 = sub_data.e1:squeeze()
            sub_data.e2 = sub_data.e2:squeeze()
        end
        sub_data.e1 = self:to_cuda(sub_data.e1)
        sub_data.e2 = self:to_cuda(sub_data.e2)
    else
        if self.params.rowEncoder == 'lookup-table' then sub_data.ep = sub_data.ep:squeeze() end
        sub_data.ep = self:to_cuda(sub_data.ep)
    end
    if self.params.colEncoder == 'lookup-table' then
        sub_data.rel = self:to_cuda(sub_data.rel):squeeze()
    else
        sub_data.seq = self:to_cuda(sub_data.seq)
        if sub_data.seq:dim() == 1 then sub_data.seq = sub_data.seq:view(sub_data.seq:size(1), 1) end
    end
    return sub_data
end

function RelationEncoderModel:load_train_data(data_file, entities)
    local train = torch.load(data_file)
    if #train > 0 then
        for i = 1, self.params.maxSeq do
            if train[i] and train[i].ep then
                train[i] = self:load_sub_data(train[i], entities)
            end
        end
    else
        self:load_sub_data(train, entities)
    end
    return train
end

function RelationEncoderModel:load_entity_data(data_file)
    return self:load_train_data(data_file, true)
end

function RelationEncoderModel:load_ep_data(data_file)
    return self:load_train_data(data_file, false)
end

function RelationEncoderModel:save_model(epoch)
    if self.params.saveModel ~= '' then
        torch.save(self.params.saveModel .. '/' .. epoch .. '-model',
            {net = self.net, encoder = self.col_encoder, col_table = self.col_table, row_table = self.row_table, opt_state = self.opt_state})
        self:tac_eval(self.params.saveModel .. '/' .. epoch, self.params.resultDir .. '/' .. epoch, self.params.evalArgs)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-ent-weights', self.params.gpuid >= 0 and self.row_table.weight:double() or self.row_table.weight)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-rel-weights', self.params.gpuid >= 0 and self.col_table.weight:double() or self.col_table.weight)
    end
end

function RelationEncoderModel:write_output()
    local function write_embeddings(f, embeddings)
        local file = io.open(f, "w")
        io.output(file)
        for i = 1, embeddings:size(1) do
            for j = 1, embeddings:size(2) do
                io.write(embeddings[i][j] .. ' ')
            end
            io.write('\n')
        end
        io.close()
    end

    -- write embeddings to file
    if self.params.saveRowEmbeddings ~= '' then
        write_embeddings(self.params.saveRowEmbeddings, self.row_table.weight)
        torch.save(self.params.saveRowEmbeddings .. '.torch', self.row_table.weight:double())
    end
    if self.params.saveColEmbeddings ~= '' then
        write_embeddings(self.params.saveColEmbeddings, self.col_table.weight)
        torch.save(self.params.saveColEmbeddings .. '.torch', self.col_table.weight:double())
    end
    require 'PatternScorer'
    if self.params.k > 0 then PatternScorer:get_top_patterns_topk(self.params.k) end
    if self.params.thresh > 0 then PatternScorer:get_top_patterns_threshold(self.params.thresh) end
end
