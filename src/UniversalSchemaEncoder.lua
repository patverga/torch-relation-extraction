--
-- User: pat
-- Date: 8/26/15
--
package.path = package.path .. ";src/?.lua"

require 'torch'
require 'rnn'
require 'optim'
require 'MatrixFactorizationModel'

local UniversalSchemaEncoder, parent = torch.class('UniversalSchemaEncoder', 'MatrixFactorizationModel')

function UniversalSchemaEncoder:build_network(pos_row_encoder, col_encoder)
    local neg_row_encoder = pos_row_encoder:clone()

    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_row_encoder)
    loading_par_table:add(col_encoder)
    loading_par_table:add(neg_row_encoder)

    -- layers to compute the dot prduct of the positive and negative samples
    local pos_dot = nn.Sequential()
    pos_dot:add(nn.NarrowTable(1, 2))
    pos_dot:add(nn.CMulTable())
    pos_dot:add(nn.Sum(2))

    local neg_dot = nn.Sequential()
    neg_dot:add(nn.NarrowTable(2, 2))
    neg_dot:add(nn.CMulTable())
    neg_dot:add(nn.Sum(2))

    -- add the parallel dot products together into one sequential network
    local net = nn.Sequential()
    net:add(loading_par_table)
    local concat_table = nn.ConcatTable()
    concat_table:add(pos_dot)
    concat_table:add(neg_dot)
    net:add(concat_table)

    -- put the networks on cuda
    self:to_cuda(net)

    -- need to do param sharing after tocuda
    pos_row_encoder:share(neg_row_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')
    return net
end


----- TRAIN -----
function UniversalSchemaEncoder:gen_training_batches(data, shuffle)
    shuffle = shuffle or true
    local batches = {}
    -- new 3 col format
    if data.num_cols or data.col then
        if #data > 0 then
            for seq_size = 1, self.params.maxSeq and math.min(self.params.maxSeq, #data) or #data do
                if data[seq_size] and data[seq_size].row then self:gen_subdata_batches_three_col(data, data[seq_size], batches, data.num_rows, shuffle) end
            end
        else  self:gen_subdata_batches_three_col(data, data, batches, data.num_rows, shuffle) end
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
        local neg_ep_batch = self:to_cuda(self:gen_neg(data, pos_ep_batch, size, max_neg))
        local rel_batch = self.params.colEncoder == 'lookup-table' and sub_data.rel:index(1, batch_indices) or sub_data.seq:index(1, batch_indices)
        local batch = { pos_ep_batch, rel_batch, neg_ep_batch}
        table.insert(batches, { data = batch, label = 1 })
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
        local neg_row_batch = self:to_cuda(self:gen_neg(data, pos_row_batch, size, max_neg))
        local col_batch = self.params.colEncoder == 'lookup-table' and sub_data.col:index(1, batch_indices) or sub_data.col_seq:index(1, batch_indices)
        local batch = { pos_row_batch, col_batch, neg_row_batch}
        table.insert(batches, { data = batch, label = 1 })
        start = start + size
    end
end

function UniversalSchemaEncoder:gen_neg(data, pos_batch, size, max_neg)
    local neg_batch
    if max_neg == 0 then
        neg_batch = pos_batch:clone():fill(1)
    elseif self.params.rowEncoder == 'lookup-table' then
        neg_batch = torch.rand(size):mul(max_neg):floor():add(1):view(pos_batch:size())
    else
        -- we want to draw a negative sample length weighted by how common that lenght is in our data
        if not self.sizes then
            local sizes = {}
            for i = 1, data.max_length do if data[i] and data[i].count then sizes[i] = data[i].count else sizes[i] = 0 end end
            self.sizes = torch.Tensor(sizes)
        end
        local neg_length = torch.multinomial(self.sizes, 1)[1]
        while (not (data[neg_length] and data[neg_length].count) or data[neg_length].count < size) do
            neg_length = torch.multinomial(self.sizes, 1)[1]
        end

        -- select 'size' random samples from the selected sequence length
        local rand_indices = {}
        while #rand_indices < size do
            table.insert(rand_indices, torch.rand(1):mul(data[neg_length].row:size(1)):floor():add(1)[1])
        end
        local rand_order = torch.LongTensor(rand_indices)
        local batch_indices = rand_order:narrow(1, 1, size)
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

        local old = true
        if(old) then
            local theta = pred[1] - pred[2]
            local prob = theta:clone():fill(1):cdiv(torch.exp(-theta):add(1))
            err = torch.log(prob):mean()
            local step = (prob:clone():fill(1) - prob)
            local df_do = { -step, step }
            net:backward(x, df_do)
        else
            self.prob_net = self.prob_net or self:to_cuda(nn.Sequential():add(nn.CSubTable()):add(nn.Sigmoid()))
            local prob =  self.prob_net:forward(pred)

            if(self.df_do) then self.df_do:resizeAs(prob) else  self.df_do = prob:clone() end

            self.df_do:copy(prob):mul(-1):add(1)
            err = prob:log():mean()

            local df_dpred = self.prob_net:backward(pred,self.df_do)
            net:backward(x,df_dpred)
        end

        if net.forget then net:forget() end
        if self.params.l2Reg > 0 then grad_params:add(self.params.l2Reg, parameters) end
        if self.params.clipGrads > 0 then
            local grad_norm = grad_params:norm(2)
            if grad_norm > self.params.clipGrads then grad_params = grad_params:div(grad_norm/self.params.clipGrads) end
        end
        if self.params.freezeRow >= epoch then self.row_table:zeroGradParameters() end
        if self.params.freezeCol >= epoch then self.col_table:zeroGradParameters() end
        return err, grad_params
    end

    optim[self.params.optimMethod](fEval, parameters, opt_config, opt_state)
    opt_config.learningRate = self.params.learningRate
    if self.params.maxNormCol > 0 then self.col_table.weight:renorm(2, 2, self.params.maxNormCol) end
    if self.params.maxNormRow > 0 then self.col_table.weight:renorm(2, 2, self.params.maxNormRow) end
    return err
end


----- Evaluate ----

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

