--
-- User: pv
-- Date: 9/21/15
--

local RelationEncoderModel = torch.class('RelationEncoderModel')


function RelationEncoderModel:init_opt()
    self.opt_config = { learningRate = self.params.learningRate, epsilon = self.params.epsilon,
        beta1 = self.params.beta1, beta2 = self.params.beta2,
        momentum = self.params.momentum, learningRateDecay = self.params.decay
    }
    self.opt_state = {}

end

--- Utils ---

function RelationEncoderModel:to_cuda(x) return self.params.gpuid >= 0 and x:cuda() or x:double() end


--- Train ---

function RelationEncoderModel:train(num_epochs)
    num_epochs = num_epochs or self.params.numEpochs
    -- optim stuff
    local parameters, gradParameters = self.net:getParameters()

    -- make sure model save dir exists
    if self.params.saveModel ~= '' then os.execute("mkdir " .. self.params.saveModel) end
    for epoch = 1, num_epochs
    do
        if (epoch % self.params.evaluateFrequency == 0) then
            self:evaluate()
            self:save_model(epoch-1)
        end
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
            epoch_error = epoch_error + self:optim_update(self.net, self.crit, example, label, parameters, gradParameters, self.opt_config, self.opt_state, epoch)

            if (i % 50 == 0) then
                io.write(string.format('\r%.2f percent complete\tspeed = %.2f examples/sec\terror = %.4f',
                    100 * i / (#batches), (i * self.params.batchSize) / (sys.clock() - startTime), (epoch_error / i)))
                io.flush()
            end
        end
        print(string.format('\nEpoch error = %f', epoch_error))
    end
    self:evaluate()
    self:save_model(num_epochs)
end


--- Evaluate ----

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


function RelationEncoderModel:tac_eval(model_file, other_args)
    if self.params.vocab ~= '' and self.params.tacYear ~= '' then
        local cmd = '/home/pat/canvas/universal-schema/univSchema/torch/tac-eval.sh ' .. self.params.tacYear .. ' ' .. model_file..'-model'
                .. ' ' .. self.params.vocab .. ' ' .. self.params.gpuid ..' ' .. self.params.maxSeq .. ' "' .. other_args .. '"' .. ' ' .. self.params.tacOut
        print(cmd)
        os.execute(cmd)
    end
end

--- IO ----

function RelationEncoderModel:load_train_data(data_file, entities)
    local train = torch.load(data_file)
    if #train > 0 then
        for i = 1, self.params.maxSeq do
            if train[i] and train[i].ep then
                if entities then
                    train[i].e1 = self:to_cuda(train[i].e1):contiguous():view(train[i].e1:size(1))
                    train[i].e2 = self:to_cuda(train[i].e2):contiguous():view(train[i].e2:size(1))
                else
                    train[i].ep = self:to_cuda(train[i].ep):contiguous():view(train[i].ep:size(1))
                end
                if self.params.testing then train[i].rel = self:to_cuda(train[i].rel):contiguous():view(train[i].rel:size(1), 1)
                else train[i].seq = self:to_cuda(train[i].seq):contiguous()
                end
            end
        end
    else
        if entities then
            train.e1 = self:to_cuda(train.e1):contiguous():view(train.e1:size(1))
            train.e2 = self:to_cuda(train.e2):contiguous():view(train.e2:size(1))
        else
            train.ep = self:to_cuda(train.ep):contiguous():view(train.ep:size(1))
        end
        if self.params.testing then train.rel = self:to_cuda(train.rel):contiguous():view(train.rel:size(1), 1)
        else train.seq = self:to_cuda(train.seq):contiguous()
        end
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
            {net = self.net, encoder = self.encoder, rel_table = self.rel_table, ent_table = self.ent_table, opt_state = self.opt_state})
        self:tac_eval(self.params.saveModel .. '/' .. epoch, self.params.otherArgs)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-ent-weights', self.params.gpuid >= 0 and self.ent_table.weight:double() or self.ent_table.weight)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-rel-weights', self.params.gpuid >= 0 and self.rel_table.weight:double() or self.rel_table.weight)
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
    if self.params.saveEpEmbeddings ~= '' then
        write_embeddings(self.params.saveEpEmbeddings, self.ent_table.weight)
        torch.save(self.params.saveEpEmbeddings .. '.torch', self.ent_table.weight:double())
    end
    if self.params.saveRelEmbeddings ~= '' then
        write_embeddings(self.params.saveRelEmbeddings, self.rel_table.weight)
        torch.save(self.params.saveRelEmbeddings .. '.torch', self.rel_table.weight:double())
    end
    require 'PatternScorer'
    if self.params.k > 0 then PatternScorer:get_top_patterns_topk(self.params.k) end
    if self.params.thresh > 0 then PatternScorer:get_top_patterns_threshold(self.params.thresh) end
end
