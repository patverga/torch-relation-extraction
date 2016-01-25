--
-- User: pat
-- Date: 9/21/15
--

require 'torch'
require 'rnn'
require 'optim'
require 'RelationEncoderModel'

local UniversalSchemaEntityEncoder, parent = torch.class('UniversalSchemaEntityEncoder', 'RelationEncoderModel')

-- TODO encoder factory dealing with different size embedding lookuptables
function UniversalSchemaEntityEncoder:build_network(pos_e1_encoder, col_encoder)
    local pos_e2_encoder = pos_e1_encoder:clone()
    local neg_e1_encoder = pos_e1_encoder:clone()
    local neg_e2_encoder = pos_e1_encoder:clone()

    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_e1_encoder)
    loading_par_table:add(pos_e2_encoder)
    loading_par_table:add(col_encoder)
    loading_par_table:add(neg_e1_encoder)
    loading_par_table:add(neg_e2_encoder)

    require 'nn-modules/ViewTable'
    self.ep_encoder = nn.Sequential()

    if self.params.compositional then
        self.ep_encoder:add(nn.ViewTable(-1, 1, self.params.rowDim))
        self.ep_encoder:add(nn.JoinTable(2))
        self.ep_encoder:add(nn.TemporalConvolution(self.params.rowDim, self.params.rowDim, 2))
        self.ep_encoder:add(nn.Tanh())
    else
        self.ep_encoder:add(nn.JoinTable(2))
    end

    -- layers to compute the dot prduct of the positive and negative samples
    local pos_ents = nn.Sequential()
    pos_ents:add(nn.NarrowTable(1, 2))
    pos_ents:add(self.ep_encoder)

    local pos_concat = nn.ConcatTable()
    pos_concat:add(pos_ents)
    pos_concat:add(nn.SelectTable(3))
    local pos_dot = nn.Sequential()
    pos_dot:add(pos_concat)
    pos_dot:add(nn.CMulTable())
    pos_dot:add(nn.Sum(2))

    local neg_ents = nn.Sequential()
    neg_ents:add(nn.NarrowTable(4, 2))
    neg_ents:add(self.ep_encoder:clone())

    local neg_concat = nn.ConcatTable()
    neg_concat:add(neg_ents)
    neg_concat:add(nn.SelectTable(3))
    local neg_dot = nn.Sequential()
    neg_dot:add(neg_concat)
    neg_dot:add(nn.CMulTable())
    neg_dot:add(nn.Sum(2))

    -- add the parallel dot products together into one sequential network
    local concat_table = nn.ConcatTable()
    concat_table:add(pos_dot)
    concat_table:add(neg_dot)

    local net = nn.Sequential()
    net:add(loading_par_table)
    net:add(concat_table)

    -- put the networks on cuda
    self:to_cuda(net)

    -- need to do param sharing after tocuda
    pos_ents:share(neg_ents, 'weight', 'bias', 'gradWeight', 'gradBias')
    neg_e1_encoder:share(neg_e2_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')
    pos_e2_encoder:share(neg_e1_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')
    pos_e1_encoder:share(pos_e2_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')

    return net
end



----- TRAIN -----
function UniversalSchemaEntityEncoder:gen_subdata_batches(sub_data, batches, max_neg, shuffle)
    local start = 1
    local rand_order = shuffle and torch.randperm(sub_data.ep:size(1)):long() or torch.range(1, sub_data.ep:size(1)):long()
    while start <= sub_data.ep:size(1) do
        local size = math.min(self.params.batchSize, sub_data.ep:size(1) - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local pos_e1_batch = sub_data.e1:index(1, batch_indices)
        local pos_e2_batch = sub_data.e2:index(1, batch_indices)
        local neg_e1_batch = self:to_cuda(torch.rand(size):mul(max_neg):floor():add(1))
        local neg_e2_batch = self:to_cuda(torch.rand(size):mul(max_neg):floor():add(1))
--        -- randomly 0 out half of the pos e1's
--        local choose_head_tail = self:to_cuda(torch.rand(size):gt(0.5):double())
--        local neg_e1_batch = pos_e1_batch:clone():cmul(choose_head_tail)
--        -- replace the 0 indicies with negative samples
--        neg_e1_batch:add(neg_ent_batch:clone():cmul(self:to_cuda(neg_e1_batch:eq(0):double())))
--        -- need to 0 out the opposites of e1
--        local neg_e2_batch = pos_e2_batch:clone():cmul(self:to_cuda(choose_head_tail:eq(0):double()))
--        neg_e2_batch:add(neg_ent_batch:clone():cmul(self:to_cuda(neg_e2_batch:eq(0):double())))


        local rel_batch = self.params.colEncoder == 'lookup-table' and sub_data.rel:index(1, batch_indices) or sub_data.seq:index(1, batch_indices)
        local batch = { pos_e1_batch, pos_e2_batch, rel_batch, neg_e1_batch, neg_e2_batch }
        table.insert(batches, { data = batch, label = 1 })
        start = start + size
    end
end

function UniversalSchemaEntityEncoder:gen_training_batches(data)
    local batches = {}
    if #data > 0 then
        for seq_size = 1, self.params.maxSeq and math.min(self.params.maxSeq, #data) or #data do
            local sub_data = data[seq_size]
            if sub_data and sub_data.ep then self:gen_subdata_batches(sub_data, batches, data.num_ents, true) end
        end
    else
        self:gen_subdata_batches(data, batches, data.num_ents, true)
    end
    return batches
end

function UniversalSchemaEntityEncoder:regularize()
    self.col_table.weight:renorm(2, 2, 3.0)
    self.row_table.weight:renorm(2, 2, 3.0)
end


function UniversalSchemaEntityEncoder:optim_update(net, criterion, x, y, parameters, grad_params, opt_config, opt_state, epoch)
    local err
    if x[2]:dim() == 1 or x[2]:size(2) == 1 then opt_config.learningRate = self.params.learningRate * self.params.kbWeight end
    local function fEval(parameters)
        if parameters ~= parameters then parameters:copy(parameters) end
        net:zeroGradParameters()
        local pred = net:forward(x)
        local theta = pred[1] - pred[2]
        local prob = theta:clone():fill(1):cdiv(torch.exp(-theta):add(1))
        err = torch.log(prob):mean()
        local step = (prob:clone():fill(1) - prob)
        local df_do = { -step, step }
        net:backward(x, df_do)
        if net.forget then net:forget() end
        if self.params.l2Reg ~= 0 then grad_params:add(self.params.l2Reg, parameters) end
        if self.params.clipGrads > 0 then
            local grad_norm = grad_params:norm(2)
            if grad_norm > self.params.clipGrads then grad_params = grad_params:div(grad_norm/self.params.clipGrads) end
        end
        if self.params.freezeEp >= epoch then self.row_table:zeroGradParameters() end
        if self.params.freezeRel >= epoch then self.col_table:zeroGradParameters() end
        return err, grad_params
    end
    optim[self.params.optimMethod](fEval, parameters, opt_config, opt_state)
    opt_config.learningRate = self.params.learningRate
    -- TODO, better way to handle this
    if self.params.regularize then self:regularize() end
    return err
end


----- Evaluate ----
function UniversalSchemaEntityEncoder:score_subdata(sub_data)
    local batches = {}
    self:gen_subdata_batches(sub_data, batches, 0, false)

    local scores = {}
    for i = 1, #batches do
        local e1_batch, e2_batch, rel_batch, _, _ = unpack(batches[i].data)
        if self.params.colEncoder == 'lookup-table' then rel_batch = rel_batch:view(rel_batch:size(1), 1) end
        if self.params.rowEncoder == 'lookup-table' then
            e1_batch = e1_batch:view(e1_batch:size(1), 1)
            e2_batch = e2_batch:view(e2_batch:size(1), 1)
        end
        local encoded_rel = self.col_encoder(self:to_cuda(rel_batch)):squeeze()
        local encoded_e1 = self.row_encoder(self:to_cuda(e1_batch)):squeeze():clone()
        local encoded_e2 = self.row_encoder(self:to_cuda(e2_batch)):squeeze()
        local encoded_ep = self.ep_encoder({encoded_e1, encoded_e2}):squeeze()
        local x = { encoded_ep, encoded_rel, }
        local score = self.cosine(x):double()
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end

