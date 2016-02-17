--
-- User: pat
-- Date: 8/26/15
--
package.path = package.path .. ";src/?.lua"

require 'torch'
require 'rnn'
require 'optim'

local UniversalSchemaEncoder = torch.class('UniversalSchemaEncoder')

--- Utils ---

function UniversalSchemaEncoder:to_cuda(x) return self.params.gpuid >= 0 and x:cuda() or x:double() end


function UniversalSchemaEncoder:__init(params, row_table, row_encoder, col_table, col_encoder, use_entities)
    self.__index = self
    self.params = params
    self.opt_config = { learningRate = self.params.learningRate, epsilon = self.params.epsilon,
        beta1 = self.params.beta1, beta2 = self.params.beta2,
        momentum = self.params.momentum, learningRateDecay = self.params.decay
    }
    self.opt_state = {}
    self.train_data = self:load_train_data(params.train, use_entities)

    -- cosine distance network for evaluation
    self.cosine = self:to_cuda(nn.CosineDistance())

    -- either load model from file or initialize new one
    if params.loadModel ~= '' then
        local loaded_model = torch.load(params.loadModel)
        self.net = self:to_cuda(loaded_model.net)
        col_encoder = self:to_cuda(loaded_model.col_encoder or loaded_model.encoder)
        row_encoder = self:to_cuda(loaded_model.row_encoder or loaded_model.row_table)
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

    -- set the criterion
    if params.criterion == 'bpr' then
        require 'nn-modules/BPRLoss'
        self.crit = nn.BPRLoss()
    elseif  params.criterion == 'hinge' then
        self.crit = nn.MarginRankingCriterion(self.params.margin)
    else
        print('Must supply option to criterion. Valid options are: bpr and hinge')
        os.exit()
    end
    self:to_cuda(self.crit)
end

--[[ a function that takes the the output of {pos_row_encoder, col_encoder, neg_row_encoder}
    and returns {pos score, neg score} ]]--
-- layers to compute the dot prduct of the positive and negative samples
function UniversalSchemaEncoder:build_scorer()
    local pos_score = nn.Sequential()
        :add(nn.NarrowTable(1, 2)):add(nn.CMulTable()):add(nn.Sum(2))
    local neg_score = nn.Sequential()
        :add(nn.NarrowTable(2, 2)):add(nn.CMulTable()):add(nn.Sum(2))
    local score_table = nn.ConcatTable()
        :add(pos_score):add(neg_score)
    return score_table
end

function UniversalSchemaEncoder:build_network(pos_row_encoder, col_encoder)
    local neg_row_encoder = pos_row_encoder:clone()

    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_row_encoder)
    loading_par_table:add(col_encoder)
    loading_par_table:add(neg_row_encoder)

    -- add the parallel dot products together into one sequential network
    local net = nn.Sequential()
        :add(loading_par_table)
        :add(self:build_scorer())

    -- put the networks on cuda
    self:to_cuda(net)

    -- need to do param sharing after tocuda
    pos_row_encoder:share(neg_row_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')
    return net
end


----- TRAIN -----

function UniversalSchemaEncoder:train(num_epochs)
    num_epochs = num_epochs or self.params.numEpochs
    -- optim stuff
    local parameters, gradParameters = self.net:getParameters()

    -- make sure model save dir exists
    if self.params.saveModel ~= '' then os.execute("mkdir -p " .. self.params.saveModel) end
    for epoch = 1, num_epochs
    do
        self.net:training()
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

function UniversalSchemaEncoder:gen_training_batches(data, shuffle)
    shuffle = shuffle or true
    local batches = {}
    -- new 3 col format
    if data.num_cols or data.col then
        if #data > 0 then
            for seq_size = 1, self.params.maxSeq and math.min(self.params.maxSeq, #data) or #data do
                if data[seq_size] and data[seq_size].row then self:gen_subdata_batches_three_col(data, data[seq_size], batches, self.row_table.weight:size(1), shuffle) end
            end
        else  self:gen_subdata_batches_three_col(data, data, batches, self.row_table.weight:size(1), shuffle) end
    else
        -- old 4 col format
        if #data > 0 then
            for seq_size = 1, self.params.maxSeq and math.min(self.params.maxSeq, #data) or #data do
                if data[seq_size] and data[seq_size].ep then self:gen_subdata_batches_four_col(data, data[seq_size], batches, data.num_eps, shuffle) end
            end
        else  self:gen_subdata_batches_four_col(data, data, batches, data.num_eps, shuffle) end
    end
    return batches
end

function UniversalSchemaEncoder:gen_subdata_batches_four_col(data, sub_data, batches, max_neg, shuffle)
    local start = 1
    local rand_order = shuffle and torch.randperm(sub_data.ep:size(1)):long() or torch.range(1, sub_data.ep:size(1)):long()
    while start <= sub_data.ep:size(1) do
        local size = math.min(self.params.batchSize, sub_data.ep:size(1) - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local pos_ep_batch = sub_data.ep:index(1, batch_indices)
        local neg_ep_batch = self:gen_neg(data, pos_ep_batch, size, max_neg)
        local rel_batch = self.params.colEncoder == 'lookup-table' and sub_data.rel:index(1, batch_indices) or sub_data.seq:index(1, batch_indices)
        local batch = { pos_ep_batch, rel_batch, neg_ep_batch}
        table.insert(batches, { data = batch, label = self:to_cuda(torch.ones(size)) })
        start = start + size
    end
end

function UniversalSchemaEncoder:gen_subdata_batches_three_col(data, sub_data, batches, max_neg, shuffle)
    local start = 1
    local rand_order = shuffle and torch.randperm(sub_data.row:size(1)):long() or torch.range(1, sub_data.row:size(1)):long()
    while start <= sub_data.row:size(1) do
        local size = math.min(self.params.batchSize, sub_data.row:size(1) - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local pos_row_batch = self.params.rowEncoder == 'lookup-table' and sub_data.row:index(1, batch_indices) or sub_data.row_seq:index(1, batch_indices)
        local neg_row_batch = self:gen_neg(data, pos_row_batch, size, max_neg)
        local col_batch = self.params.colEncoder == 'lookup-table' and sub_data.col:index(1, batch_indices) or sub_data.col_seq:index(1, batch_indices)
        local batch = {pos_row_batch, col_batch, neg_row_batch}
        table.insert(batches, { data = batch, label = self:to_cuda(torch.ones(size)) })
        start = start + size
    end
end

function UniversalSchemaEncoder:gen_neg(data, pos_batch, neg_sample_count, max_neg)
    local neg_batch
    if max_neg == 0 then
        neg_batch = pos_batch:clone():fill(0)
    elseif self.params.rowEncoder == 'lookup-table' then
        neg_batch = torch.rand(neg_sample_count):mul(max_neg):floor():add(1):view(pos_batch:size())
    else
        -- we want to draw a negative sample length weighted by how common that lenght is in our data
        if not self.sizes then
            local sizes = {}
            for i = 1, data.max_length do if data[i] and data[i].count then sizes[i] = data[i].count else sizes[i] = 0 end end
            self.sizes = torch.Tensor(sizes)
        end
        local neg_length = torch.multinomial(self.sizes, 1)[1]
        while (not (data[neg_length] and data[neg_length].count) or data[neg_length].count < neg_sample_count) do
            neg_length = torch.multinomial(self.sizes, 1)[1]
        end

        -- select 'size' random samples from the selected sequence length
        local rand_indices = {}
        while #rand_indices < neg_sample_count do
            table.insert(rand_indices, torch.rand(1):mul(data[neg_length].row:size(1)):floor():add(1)[1])
        end
        local rand_order = torch.LongTensor(rand_indices)
        local batch_indices = rand_order:narrow(1, 1, neg_sample_count)
        neg_batch = data[neg_length].row_seq:index(1, batch_indices)
    end
    return neg_batch
end

function UniversalSchemaEncoder:optim_update(net, criterion, x, y, parameters, grad_params, opt_config, opt_state, epoch)
    local err
    if x[2]:dim() == 1 or x[2]:size(2) == 1 then opt_config.learningRate = self.params.learningRate * self.params.kbWeight end
    local function fEval(parameters)
        if parameters ~= parameters then parameters:copy(parameters) end
        net:zeroGradParameters()
        local pred = net:forward(x)
        err = criterion:forward(pred, y)
        local df_do = criterion:backward(pred, y)
        net:backward(x, df_do)

        if net.forget then net:forget() end
        if self.params.l2Reg > 0 then grad_params:add(self.params.l2Reg, parameters) end
        if self.params.clipGrads > 0 then
            local grad_norm = grad_params:norm(2)
            if grad_norm > self.params.clipGrads then grad_params = grad_params:div(grad_norm/self.params.clipGrads) end
        end
        if self.params.freezeRow >= epoch then self.row_encoder:zeroGradParameters() end
        if self.params.freezeCol >= epoch then self.col_encoder:zeroGradParameters() end
        return err, grad_params
    end

    optim[self.params.optimMethod](fEval, parameters, opt_config, opt_state)
    opt_config.learningRate = self.params.learningRate
    if self.params.maxNormCol > 0 then self.col_table.weight:renorm(2, 2, self.params.maxNormCol) end
    if self.params.maxNormRow > 0 then self.row_table.weight:renorm(2, 2, self.params.maxNormRow) end
    return err
end



----- Evaluate ----

function UniversalSchemaEncoder:evaluate()
    self.net:evaluate()
    if self.params.test ~= '' then
        self:map(self.params.test, true)
    end
    self.net:training()
end

function UniversalSchemaEncoder:map(fileStr, high_score)
    print('Calculating MAP')
    local map = 0.0
    local file_count = 0
    for file in string.gmatch(fileStr, "[^,]+") do
        local ap = self:avg_precision(file, high_score) * 100
        map = map + ap
        file_count = file_count + 1
        io.write(string.format('\rcurrent map : %2.3f \t last ap : %2.3f\t%s',
            (map / math.max(file_count, 1)), ap, file)); io.flush()
    end
    map = map / math.max(1.0, file_count)
    print(string.format('\nMAP : %2.3f', map))
end

function UniversalSchemaEncoder:avg_precision(file, high_score)
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

function UniversalSchemaEncoder:score_test_data(data)
    local scores = {}
    local labels = {}
    if #data > 0 or data.ep == nil then
        for _, dat in pairs(data) do
            if dat and torch.type(dat) ~= 'number' and (dat.ep or dat.row_seq) then
                local score, label = self:score_subdata(dat, scores, labels)
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


function UniversalSchemaEncoder:score_subdata(sub_data)
    local batches = {}
    if sub_data.ep then self:gen_subdata_batches_four_col(sub_data, sub_data, batches, 0, false)
    else self:gen_subdata_batches_three_col(sub_data, sub_data, batches, 0, false) end
    local scores = {}
    for i = 1, #batches do
        local row_batch, col_batch, _ = unpack(batches[i].data)
        if self.params.colEncoder == 'lookup-table' then col_batch = col_batch:view(col_batch:size(1), 1) end
        if self.params.rowEncoder == 'lookup-table' then row_batch = row_batch:view(row_batch:size(1), 1) end
        local encoded_rel = self.col_encoder(self:to_cuda(col_batch)):squeeze():clone()
        local encoded_ent = self.row_encoder(self:to_cuda(row_batch)):squeeze()
        local x = { encoded_rel, encoded_ent }
        local score = self.cosine(x):double()
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end


function UniversalSchemaEncoder:tac_eval(model_file, out_dir, eval_args)
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

function UniversalSchemaEncoder:load_train_data(data_file, entities)
    local train = torch.load(data_file)
    -- new 3 col format
    if train.num_cols then
        if #train > 0 then
            for i = 1, self.params.maxSeq do
                if train[i] and train[i].row then train[i] = self:load_sub_data_three_col(train[i]) end
            end
        else
            self:load_sub_data_four_col(train, entities)
        end
    else
        -- old 4 col format
        if #train > 0 then
            for i = 1, self.params.maxSeq do
                if train[i] and train[i].ep then train[i] = self:load_sub_data_four_col(train[i], entities) end
            end
        else  self:load_sub_data_four_col(train, entities) end
    end
    return train
end

function UniversalSchemaEncoder:load_sub_data_four_col(sub_data, entities)
    if entities then
        if self.params.rowEncoder == 'lookup-table' then
            sub_data.e1 = sub_data.e1:squeeze()
            sub_data.e2 = sub_data.e2:squeeze()
        end
    else
        if self.params.rowEncoder == 'lookup-table' and sub_data.ep:dim() > 1 and sub_data.ep:size(1) > 1
        then sub_data.ep = sub_data.ep:squeeze() end
    end
    if self.params.colEncoder == 'lookup-table' then
        if self.params.relationPool == '' then sub_data.rel = sub_data.rel:squeeze()
        else sub_data.rel = sub_data.rel:view(sub_data.rel:size(1), sub_data.rel:size(2)) end
    else
        if sub_data.seq:dim() == 1 then sub_data.seq = sub_data.seq:view(sub_data.seq:size(1), 1) end
    end
    return sub_data
end

function UniversalSchemaEncoder:load_sub_data_three_col(sub_data, entities)
    if self.params.rowEncoder == 'lookup-table' and sub_data.row:dim() > 1 and sub_data.row:size(1) > 1
    then
        sub_data.row = sub_data.row:squeeze()
    else
        sub_data.row_seq = self:to_cuda(sub_data.row_seq)
        if sub_data.row_seq:dim() == 1 then sub_data.row_seq = sub_data.row_seq:view(sub_data.row_seq:size(1), 1) end
    end
    if self.params.colEncoder == 'lookup-table' and self.params.relationPool == '' and sub_data.col:dim() > 1 and sub_data.col:size(1) > 1
    then
        sub_data.col = sub_data.col:squeeze()
    else
        if sub_data.col_seq:dim() == 1 then sub_data.col_seq = sub_data.col_seq:view(sub_data.col_seq:size(1), 1)
        elseif self.params.modelType == 'max' and sub_data.col_seq:dim() == 2 then
            sub_data.col_seq = sub_data.col_seq:view(sub_data.col_seq:size(1), 1, sub_data.col_seq:size(2))
        end
    end
    return sub_data
end


function UniversalSchemaEncoder:save_model(epoch)
    if self.params.saveModel ~= '' then
        self.net:clearState()
        local cpu_opt = {}
        for k,v in pairs(self.opt_state) do cpu_opt[k] = torch.type(v) == 'torch.CudaTensor' and v:double() or v end

        torch.save(self.params.saveModel .. '/' .. epoch .. '-model',
            {net = self.net:clone():float(), col_encoder = self.col_encoder:clone():float(), row_encoder = self.row_encoder:clone():float(), opt_state = cpu_opt})
        self:tac_eval(self.params.saveModel .. '/' .. epoch, self.params.resultDir .. '/' .. epoch, self.params.evalArgs)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-rows', self.params.gpuid >= 0 and self.row_table.weight:double() or self.row_table.weight)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-cols', self.params.gpuid >= 0 and self.col_table.weight:double() or self.col_table.weight)
    end
end

function UniversalSchemaEncoder:write_embeddings_to_txt()
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
end
