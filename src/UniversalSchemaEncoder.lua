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
    self.correct_label = 1
    self.params = params
    self.opt_config = { learningRate = self.params.learningRate, epsilon = self.params.epsilon,
        beta1 = self.params.beta1, beta2 = self.params.beta2,
        momentum = self.params.momentum, learningRateDecay = self.params.decay
    }
    self.opt_state = {}
    self.train_data = self:load_train_data(params.train, use_entities)
    self.label_size = self.train_data.num_labels or 41 -- todo get from data

    -- cosine distance network for evaluation
    self.cosine = self:to_cuda(nn.CosineDistance())

    -- either load model from file or initialize new one
    if params.loadModel ~= '' then
        local loaded_model = torch.load(params.loadModel)
        self.net = self:to_cuda(loaded_model.net)
        col_encoder = self:to_cuda(loaded_model.col_encoder or loaded_model.encoder)
        row_encoder = self:to_cuda(loaded_model.row_encoder or loaded_model.row_table)
        col_table = col_encoder:findModules('nn.LookupTable')[1] or col_encoder:findModules('LookupTableMaskPad')[1] -- assumes the encoder has exactly 1 lookup table
        row_table = row_encoder:findModules('nn.LookupTable')[1] or row_encoder:findModules('LookupTableMaskPad')[1]
        self.opt_state = loaded_model.opt_state
        for key, val in pairs(loaded_model.opt_state) do
            if (torch.type(val) == 'torch.DoubleTensor') then self.opt_state[key] = self:to_cuda(val) end
        end
    else
        self.net = self:build_network(row_encoder, col_encoder)
    end
    self.row_table = row_table
    self.col_table = col_table
    self.col_encoder = col_encoder
    self.row_encoder = row_encoder

    -- set the criterion
    if params.criterion == 'bpr' then self.crit = nn.BPRLoss();
        if params.negSamples > 1 then
            print('Warning : negSamples set to ' .. params.negSamples ..'. BPR requires 1 negative sample, only using 1.')
            params.negSamples = 1
        end
    elseif params.criterion == 'hinge' then self.crit = nn.MarginCriterion(self.params.margin)
    elseif params.criterion == 'bce' then self.crit = nn.BCECriterion()
    elseif params.criterion == 'cross-entropy' then self.crit = nn.CrossEntropyCriterion()
    elseif params.criterion == 'multi-margin' then self.crit = nn.MultiMarginCriterion(self.params.p, nil, self.params.margin)
    elseif params.criterion == 'kl-divergence' then self.crit = nn.DistKLDivCriterion()
    else print('Must supply option to criterion. Valid options are: bpr, bce and hinge'); os.exit()
    end
    self:to_cuda(self.crit)
    self:regularize_hooks()
end


--[[ a function that takes the the output of {pos_row_encoder, col_encoder, neg_row_encoder}
    and returns {pos score, neg score} ]]--
function UniversalSchemaEncoder:build_scorer()
    -- layers to compute the dot prduct of the positive and negative samples
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
--    self:evaluate(0)
    if num_epochs == -1 and self.params.test == '' and self.params.fb15kDir == '' and self.params.vocab == '' then
        print('If -numEpochs = -1, must supply -test, -fb15kDir, or -vocab and -tacYear'); os.exit()
    end

    -- make sure model save dir exists
    if self.params.resultDir ~= '' then os.execute("mkdir -p " .. self.params.resultDir) end
    local epoch, best_eval_score, decrease_epochs = 1, -1, 0
    -- either train until convergence or until epecified number of epochs
    while num_epochs == -1 or epoch <= num_epochs
    do
        self.net:training()
        local startTime = sys.clock()
        io.write('\nGenerating batches ... '); io.flush()
        local batches = self:gen_training_batches(self.train_data)
        local shuffle = torch.randperm(#batches)
        local epoch_error = 0
        io.write('\rStarting epoch '.. epoch ..(num_epochs == -1 and '. Training until convergence.' or ' of ' .. num_epochs) ..'\n'); io.flush()
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
        print(string.format('\nEpoch error = %f\n', epoch_error))
        if (epoch % self.params.evaluateFrequency == 0 and (num_epochs == -1 or epoch < num_epochs)) then
            local eval_score = self:evaluate(epoch)
            if eval_score == -1 then self:save_model(epoch)
            elseif eval_score >= best_eval_score then self:save_model('best'); best_eval_score = eval_score; decrease_epochs = 0
            else
                decrease_epochs = decrease_epochs+1
                if num_epochs == -1 and decrease_epochs >= self.params.maxDecreaseEpochs then num_epochs = epoch -- converged- stop training
                else print('Score decreased for ' .. decrease_epochs ..' epoch out of max ' .. self.params.maxDecreaseEpochs) end
            end
        end
        epoch = epoch + 1
    end
    if epoch-1 % self.params.evaluateFrequency ~= 0 then
        local eval_score = self:evaluate(num_epochs)
        if eval_score == -1 then self:save_model(num_epochs)
        elseif eval_score > best_eval_score then self:save_model('best')
        end
    end
end

function UniversalSchemaEncoder:optim_update(net, criterion, x, y, parameters, grad_params, opt_config, opt_state, epoch)
    local err
--    if self.params.kbWeight ~= 1 and x[2]:dim() == 1 or x[2]:size(2) == 1 then opt_config.learningRate = opt_config.learningRate * self.params.kbWeight end
    local function fEval(parameters)
        if self.params.criterion ~= 'bce' and self.params.criterion ~= 'kl-divergence' then y = y:view(-1) end
        y = self:to_cuda(y)
        if torch.type(x) == 'table' then for i = 1, #x do x[i] = self:to_cuda(x[i]) end else x = self:to_cuda(x) end
        if self.params.relationPool ~= '' and self.params.colEncoder ~= 'lookup-table' and x[2]:dim() < 3 then x[2] = x[2]:view(-1, 1, x[2]:size(2)) end
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
        y = y:double()
        if torch.type(x) == 'table' then for i = 1, #x do x[i] = x[i]:double() end else x = x:double() end
        return err, grad_params
    end

    optim[self.params.optimMethod](fEval, parameters, opt_config, opt_state)
    self:regularize_hooks()
    opt_config.learningRate = self.params.learningRate
    return err
end

function UniversalSchemaEncoder:regularize_hooks()
    if self.params.colNorm > 0 then self.col_table.weight:renorm(2, 1, self.params.colNorm) end
    if self.params.rowNorm > 0 then self.row_table.weight:renorm(2, 1, self.params.rowNorm) end
end

function UniversalSchemaEncoder:gen_training_batches(data, shuffle)
    shuffle = shuffle or true
    local batches = {}
    -- new 3 col format
    if data.num_cols or data.col then
        for seq_size, sub_data in pairs(data) do
            if torch.type(sub_data) == 'table' and seq_size <= self.params.maxSeq and sub_data.row then
                local neg_size = self.row_table and self.row_table.weight:size(1) or 1
                self:gen_subdata_batches_three_col(data, sub_data, batches, neg_size, shuffle)
            end
        end
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
        local neg_ep_batch = pos_ep_batch:clone():fill(0):view(-1,1)
        if max_neg > 0 then
            neg_ep_batch = self:gen_neg(data, pos_ep_batch, size, max_neg)
            for i = 2, self.params.negSamples do
                neg_ep_batch = neg_ep_batch:cat(self:gen_neg(data, pos_ep_batch, size, max_neg))
            end
        end
        local rel_batch = self.params.colEncoder == 'lookup-table' and sub_data.rel:index(1, batch_indices) or sub_data.seq:index(1, batch_indices)
        local batch = { pos_ep_batch, rel_batch, neg_ep_batch}
        local target_tensor = torch.ones(size)
        if self.params.criterion == 'bce' or self.params.criterion == 'kl-divergence' then
            local a = target_tensor:view(-1,1)
            local b = torch.Tensor(target_tensor:size(1), self.params.negSamples):fill(0)
            target_tensor = a:cat(b)
        end
        table.insert(batches, { data = batch, label = target_tensor })
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
        local neg_row_batch = pos_row_batch:clone():fill(0):view(-1,1)
        if max_neg > 0 then
            neg_row_batch = self:gen_neg(data, pos_row_batch, size, max_neg)
            for i = 2, self.params.negSamples do
                neg_row_batch = neg_row_batch:cat(self:gen_neg(data, pos_row_batch, size, max_neg))
            end
        end
        local col_batch = self.params.colEncoder == 'lookup-table' and sub_data.col:index(1, batch_indices) or sub_data.col_seq:index(1, batch_indices)
        local batch = {pos_row_batch, col_batch, neg_row_batch }
        local target_tensor = torch.ones(size)
        if self.params.criterion == 'bce' or self.params.criterion == 'kl-divergence' then
            local a = target_tensor:view(-1,1)
            local b = torch.Tensor(target_tensor:size(1), self.params.negSamples):fill(0)
            target_tensor = a:cat(b)
        end
        table.insert(batches, { data = batch, label = target_tensor })
        start = start + size
    end
end

function UniversalSchemaEncoder:gen_neg(data, pos_batch, neg_sample_count, max_neg)
    local neg_batch
    if max_neg == 0 then
        neg_batch = pos_batch:clone():fill(0)
    elseif self.params.typeSampleFile ~= '' then return self:type_sampling(data, pos_batch, neg_sample_count, max_neg)
    elseif self.params.rowEncoder == 'lookup-table' then
        neg_batch = torch.rand(neg_sample_count):mul(max_neg):floor():add(1):view(pos_batch:size()):view(-1,1)
        -- filter false positives
        local r = 1
        local replace = neg_batch:eq(pos_batch):sum()
        if replace  > 0 then
            local rands = torch.rand(neg_sample_count):mul(max_neg):floor():add(1):view(pos_batch:size()):view(-1,1)
            for i = 1, neg_batch:size(1) do
                while neg_batch[i] == pos_batch[i] do
                    neg_batch[i] = rands[r]; r = r+1
                    if r > rands:size(1) then
                        rands = torch.rand(neg_sample_count):mul(max_neg):floor():add(1):view(pos_batch:size()):view(-1,1); r =1
                    end
                end
            end
        end

--        neg_batch = torch.randperm(max_neg):view(1,-1):expandAs(torch.Tensor(pos_batch:size(1),max_neg))
--        local pos_expanded = pos_batch:view(-1,1):expandAs(neg_batch)
--        neg_batch = neg_batch:maskedSelect(neg_batch:ne(pos_expanded)):view(pos_batch:size(1), max_neg-1)
--        neg_batch = neg_batch:narrow(2,1,self.params.negSamples)
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

function UniversalSchemaEncoder:type_sampling(data, pos_batch, neg_sample_count, max_neg)
    -- load type map
    if not self.type_sample_map then
        print('\nLoading type sample map')
        self.type_sample_map = {}
        local line_num = 0
        for line in io.lines(self.params.typeSampleFile) do
            local ep, ep_replacements = string.match(line, "([^\t]+)\t([^\t]+)")
            ep = tonumber(ep)
            if not self.type_sample_map[ep] then self.type_sample_map[ep] = {} end
            for ep_2 in string.gmatch(ep_replacements, "[^,]+") do
                table.insert(self.type_sample_map[ep], tonumber(ep_2))
            end
            self.type_sample_map[ep] = torch.Tensor(self.type_sample_map[ep])
            line_num = line_num+1;  if line_num % 1000 == 0 then io.write('\rline : ' .. line_num); io.flush() end
        end
        print('\nDone')
    end
    -- choose negative samples
    local neg_batch = pos_batch.new(pos_batch:size())
    for i = 1, pos_batch:size(1) do
        local ep = pos_batch[i]
        local replacements = self.type_sample_map[ep]
        if replacements then
            local rand = torch.rand(1):mul(replacements:size(1)):floor():add(1)[1]
            neg_batch[i] = replacements[rand]
        else
            neg_batch[i] = torch.rand(1):mul(max_neg):floor():add(1)
        end
    end
    return neg_batch:view(-1,1)
end



----- Evaluate ----

function UniversalSchemaEncoder:evaluate(epoch)
    self.net:evaluate()
    -- increase  batch size for eval
    local train_batch_size = self.params.batchSize
    self.params.batchSize = 10000
    local map = self.params.test ~= '' and self:map(self.params.test, true) or -1
    local accuracy = self.params.accuracyTest ~= '' and self:accuracy(self.params.accuracyTest) or -1
    local mrr, hits_at_10 = self.params.fb15kDir ~= '' and self:fb15k_evluation(self.params.fb15kDir, true) or -1, -1
    local F1 = (self.params.vocab ~= '' and self.params.tacYear ~= '')
            and self:tac_eval(self.params.resultDir .. '/' .. epoch, self.params.resultDir .. '/' .. epoch, self.params.evalArgs)
            or -1
    self.net:clearState()
    self.params.batchSize = train_batch_size
    return math.max(math.max(math.max(map, accuracy), (mrr + hits_at_10) ), F1)
end

function UniversalSchemaEncoder:map(dirStr, high_score)
    print('Calculating MAP')
    local map = 0.0
    local file_count = 0
    for file in io.popen('ls ' .. dirStr):lines() do
        local ap = self:avg_precision(dirStr ..'/' .. file, high_score)
        if ap then
            ap = ap*100
            map = map + ap
            file_count = file_count + 1
            io.write(string.format('\rcurrent map : %2.3f \t last ap : %2.3f\t%s',
                (map / math.max(file_count, 1)), ap, file)); io.flush()
        else
            print('Failed for file : '..file )
        end
    end
    map = map / math.max(1.0, file_count)
    print(string.format('\nMAP : %2.3f\n', map))
    return map
end

function UniversalSchemaEncoder:avg_precision(file, high_score)
    local data = torch.load(file)
    -- score each of the test samples
    local scores, labels = self:score_test_data(data)
    -- sort scores and attach the original labels correctly
    local ap
    if scores and scores:dim() > 0 then
        local sorted_scores, sorted_idx = torch.sort(scores, 1, high_score)
        local sorted_labels = labels:index(1, sorted_idx)
        -- TODO : stupid calc, use torch
        ap = 0.0
        local pos_n = 0
        -- iterate over all the scored data
        for rank = 1, sorted_labels:size(1) do
            -- if label is true, increment positive count and update avg p
            if (sorted_labels[rank] == self.correct_label) then
                ap = ((ap * pos_n) / (pos_n + 1)) + (1.0 / rank)
                pos_n = pos_n + 1
            end
        end
    end
    return ap
end

function UniversalSchemaEncoder:fb15k_evluation(dir, high_score)
    local mrr = 0
    local hits_at_10 = 0
    local count = 0
    local example_results = {}
    for file in io.popen('ls ' .. dir):lines() do
        local data = torch.load(dir..'/'..file)
        if self.params.filterSingleRelations then
            local data_filtered = {}
            for example_idx, example_data in pairs(data) do
                if example_data and example_data[1] and example_data[1].label:sum() == 0 then
                    example_data[1] = nil
                    table.insert(data_filtered, example_data)
                end
            end
            data = data_filtered
        end
        for i, sub_data in pairs(data) do
            if self.params.evalCutoff == 0 or count < self.params.evalCutoff then
                local num_relations_of_positive = 1
                local scores, labels
                if not sub_data.label then
                    local all_scores, all_labels = {}, {}
                    for num_relations, sub_sub_data in pairs(sub_data) do
                        -- score each of the test samples
                        if self.row_encoder ~= 'lookup-table' then
                            if sub_data.row_seq and sub_data.row_seq then sub_data.row_seq = sub_data.row_seq:view(-1,1) end
                        end
                        local s, l = self:score_subdata(sub_sub_data)
                        -- there should be exactly 1 positive example - figure out how many relations it has for log
                        if l:sum() == 1 then num_relations_of_positive = num_relations end
                        table.insert(all_scores, nn.JoinTable(1)(s))
                        table.insert(all_labels, l)
                    end
                    scores = nn.JoinTable(1)(all_scores)
                    labels = nn.JoinTable(1)(all_labels)
                else
                    scores, labels = self:score_subdata(sub_data)
                    scores = nn.JoinTable(1)(scores)
                end
                -- sort scores and attach the original labels correctly
                local sorted_scores, sorted_idx = torch.sort(scores, 1, high_score)
                local sorted_labels = labels:index(1, sorted_idx:view(-1))
                local rank = sorted_labels:nonzero()
--                rank = rank:dim() > 0 and rank[1][1] or (torch.rand(1):mul(sorted_labels:size(1)):floor()+1)[1]
                if rank:dim() > 0 then rank = rank[1][1] else print ('rank fail ', rank, i) end
                hits_at_10 = rank <= 10 and hits_at_10 + 1 or hits_at_10
                count = count + 1
                mrr = mrr + (1/rank)
                -- keep track of stats for log
                table.insert(example_results, num_relations_of_positive .. '\t' .. rank .. '\t' .. sorted_labels:size(1) .. '\n')
                io.write(string.format('\rmrr : %2.3f \t hits@10 : %2.3f \tnum: %d', (mrr/count)*100, (hits_at_10/count)*100, count)); io.flush()
            end
        end
    end
    mrr = (mrr / count) * 100
    hits_at_10 = (hits_at_10 / count) * 100
    print ('\nMRR: ' .. mrr .. '\t HITS@10: ' .. hits_at_10)

    if self.params.evalLog ~= '' then
        local file = io.open(self.params.evalLog, "w"); io.output(file)
        for _, line in pairs(example_results) do io.write(line) end
        io.close()
    end

    return mrr, hits_at_10
end

function UniversalSchemaEncoder:accuracy(file)
    print('Accuracy not implemented for this model.'); os.exit()
end

function UniversalSchemaEncoder:score_test_data(data)
    local scores = {}
    local labels = {}
    if #data > 0 or data.ep == nil then
        --TODO hack
        if self.params.relationPool ~= '' then data = data[1] end
        for _, dat in pairs(data) do
            if dat and torch.type(dat) == 'table' and (dat.ep or dat.row_seq) then
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
    if scores and scores:dim()> 0 then scores = scores:view(scores:size(1)) end
    return scores, labels
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
    require 'eval/TacEvalCmdArgs'
    require 'eval/ScoringFunctions'
    local tac_eval_str = eval_args:gsub('"','')
    local scored_candidate = os.tmpname()
    local arg_table = {
        [1]='-vocabFile', [2]=self.params.vocab,
        [3]='-maxSeq', [4]=self.params.maxSeq,
        [5]='-outFile', [6]=scored_candidate,
        [7]='-gpuid', [8]=self.params.gpuid
    }
    for arg in string.gmatch(tac_eval_str, "[^,]+") do table.insert(arg_table, arg) end
    local tac_params = TacEvalCmdArgs:parse(arg_table)

    local kb_encoder, text_encoder, net = self.row_encoder, self.col_encoder, self.net
    self.net:evaluate(); kb_encoder:evaluate(); text_encoder:evaluate()
    local scorer
    if tac_params.scoringType == 'cosine' then
        -- directly compare column representations
        kb_encoder = text_encoder:clone()
        scorer = CosineSentenceScorer(tac_params, self.net, kb_encoder, text_encoder)
    elseif tac_params.scoringType == 'classifier' then
        scorer = SentenceClassifier(tac_params, self.net, kb_encoder, text_encoder)
    elseif tac_params.scoringType == 'pool-classifier' then
        scorer = PoolSentenceClassifier(tac_params, self.net, kb_encoder, text_encoder)
    elseif tac_params.scoringType == 'network' then
        scorer = NetworkScorer(tac_params, self.net, kb_encoder, text_encoder)
    else
        print('Must supply valid scoringType : cosine, network, classifier, pool-classifier')
        os.exit()
    end
    scorer:run()
    if self.replicate then self.replicate.nfeatures = self.params.negSamples+1 end
    --    os.execute("mkdir -p " .. model_file)
    local cmd = '${TH_RELEX_ROOT}/bin/tac-evaluation/tune-thresh-prescored.sh ' .. self.params.tacYear ..
            ' ' .. scored_candidate .. ' ' .. out_dir

    -- get F1 returned by the tune script which is the last token
    print('Tuning per relation thresholds ...')
    local handle = io.popen(cmd)
    local F1
    for token in handle:read("*a"):gmatch("%S+") do
        F1 = token; io.write(token .. ' ')
    end
    handle:close(); io.flush()
    print (tonumber(F1))
    return tonumber(F1)
end





--- IO ----

function UniversalSchemaEncoder:load_train_data(data_file, entities)
    local train = torch.load(data_file)
    -- new 3 col format
    if train.num_cols then
        for seq_len, sub_data in pairs(train) do
            if torch.type(sub_data) == 'table' and sub_data and sub_data.row then
                train[seq_len] = self:load_sub_data_three_col(sub_data)
            end
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
        if self.params.rowEncoder == 'lookup-table' and sub_data.ep:dim() > 1 then sub_data.ep = sub_data.ep:view(-1) end
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
    if self.params.rowEncoder == 'lookup-table' then
        if sub_data.row:dim() > 1 then sub_data.row = sub_data.row:view(-1) end
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


function UniversalSchemaEncoder:save_model(model_name)
    if self.params.saveModel and self.params.resultDir ~= '' then
        self.net:clearState()
        collectgarbage()
        if self.params.noCloneSave then
            torch.save(self.params.resultDir .. '/' .. model_name .. '-model',
                {net = self.net, col_encoder = self.col_encoder, row_encoder = self.row_encoder, opt_state = self.opt_state})
            if self.row_table then torch.save(self.params.resultDir .. '/' .. model_name .. '-rows', self.row_table.weight) end
            if self.col_table then torch.save(self.params.resultDir .. '/' .. model_name .. '-cols', self.col_table.weight) end
        else
            local cpu_opt = {}
            for k,v in pairs(self.opt_state) do cpu_opt[k] = torch.type(v) == 'torch.CudaTensor' and v:double() or v end
            local cpu_net = self.params.gpuid >= 0 and self.net:clone():float() or self.net; collectgarbage()
            local cpu_col_encoder = self.params.gpuid >= 0 and self.col_encoder:clone():float() or self.col_encoder; collectgarbage()
            local cpu_row_encoder = self.params.gpuid >= 0 and self.row_encoder:clone():float() or self.row_encoder; collectgarbage()
            torch.save(self.params.resultDir .. '/' .. model_name .. '-model',
                {net = cpu_net, col_encoder = cpu_col_encoder, row_encoder = cpu_row_encoder, opt_state = cpu_opt})
            if self.row_table then torch.save(self.params.resultDir .. '/' .. model_name .. '-rows', self.params.gpuid >= 0 and self.row_table.weight:clone():float() or self.row_table.weight) end
            if self.col_table then torch.save(self.params.resultDir .. '/' .. model_name .. '-cols', self.params.gpuid >= 0 and self.col_table.weight:clone():float() or self.col_table.weight) end
        end
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
