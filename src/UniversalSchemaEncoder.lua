--
-- User: pat
-- Date: 8/26/15
--
package.path = package.path .. ";src/?.lua"

require 'torch'
require 'rnn'
require 'optim'
require 'RelationEncoderModel'

local UniversalSchemaEncoder, parent = torch.class('UniversalSchemaEncoder', 'RelationEncoderModel')

function UniversalSchemaEncoder:__init(params, ent_table, ent_encoder, rel_table, rel_encoder)
    self.__index = self
    self.params = params
    self:init_opt()
    self.squeeze_rel = params.relations or false
    self.train_data = self:load_ep_data(params.train)

    -- cosine distance network for evaluation
    self.cosine = self:to_cuda(nn.CosineDistance())

    -- either load model from file or initialize new one
    if params.loadModel ~= '' then
        local loaded_model = torch.load(params.loadModel)
        self.net = self:to_cuda(loaded_model.net)
        rel_encoder = self.net:get(1):get(2) --self:to_cuda(loaded_model.encoder)
        ent_table = self.net:get(1):get(1) --self:to_cuda((loaded_model.ent_table or self.net:get(1):get(1)))
        rel_table = self:to_cuda(loaded_model.rel_table)
        self.opt_state = loaded_model.opt_state
        for key, val in pairs(loaded_model.opt_state) do if (torch.type(val) == 'torch.DoubleTensor') then self.opt_state[key] = self:to_cuda(val) end; end
    else
        self.net = self:build_network(ent_encoder, rel_encoder)
    end
    self.ent_table = ent_table
    self.rel_table = rel_table
    self.rel_encoder = rel_encoder
    self.ent_encoder = ent_encoder
end


function UniversalSchemaEncoder:build_network(pos_ent_encoder, rel_encoder)
    local neg_ent_encoder = pos_ent_encoder:clone()

    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_ent_encoder)
    loading_par_table:add(rel_encoder)
    loading_par_table:add(neg_ent_encoder)

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
    pos_ent_encoder:share(neg_ent_encoder, 'weight', 'bias', 'gradWeight', 'gradBias')
    return net
end


----- TRAIN -----
function UniversalSchemaEncoder:gen_subdata_batches(sub_data, batches, max_neg, shuffle)
--    shuffle = shuffle or true
    local start = 1
    local rand_order = shuffle and torch.randperm(sub_data.ep:size(1)):long() or torch.range(1, sub_data.ep:size(1)):long()
    while start <= sub_data.ep:size(1) do
        local size = math.min(self.params.batchSize, sub_data.ep:size(1) - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local pos_ep_batch = sub_data.ep:index(1, batch_indices)
        local neg_ep_batch = self:to_cuda(torch.rand(size):mul(max_neg):floor():add(1)):view(pos_ep_batch:size())
        local rel_batch = self.params.encoder == 'lookup-table' and sub_data.rel:index(1, batch_indices) or sub_data.seq:index(1, batch_indices)
--        if self.squeeze_rel then rel_batch = rel_batch:squeeze() end
        local batch = { pos_ep_batch, rel_batch, neg_ep_batch}
        table.insert(batches, { data = batch, label = 1 })
        start = start + size
    end
end


function UniversalSchemaEncoder:gen_training_batches(data)
    local batches = {}
    if #data > 0 then
        for seq_size = 1, self.params.maxSeq and math.min(self.params.maxSeq, #data) or #data do
            local sub_data = data[seq_size]
            if sub_data and sub_data.ep then self:gen_subdata_batches(sub_data, batches, data.num_eps, true) end
        end
    else
        self:gen_subdata_batches(data, batches, data.num_eps, true)
    end
    return batches
end


function UniversalSchemaEncoder:regularize()
    self.rel_table.weight:renorm(2, 2, 3.0)
    --    self.ent_table.weight:renorm(2, 2, 3.0)
end


function UniversalSchemaEncoder:optim_update(net, criterion, x, y, parameters, grad_params, opt_config, opt_state, epoch)
    local err
    if x[2]:dim() == 1 or x[2]:size(2) == 1 then opt_config.learningRate = self.params.learningRate * self.params.kbWeight end
    local function fEval(parameters)
        if parameters ~= parameters then parameters:copy(parameters) end
        net:zeroGradParameters()
--        x = {x[1], x[2]:view(x[2]:size(1), 1), x[3]}
--        print(x)
--        print(x[1]:max(), x[2]:max(), x[3]:max())

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
        if self.params.freezeEp >= epoch then self.ent_table:zeroGradParameters() end
        if self.params.freezeRel >= epoch then self.rel_table:zeroGradParameters() end
        return err, grad_params
    end

    optim[self.params.optimMethod](fEval, parameters, opt_config, opt_state)
    opt_config.learningRate = self.params.learningRate
    -- TODO, better way to handle this
    if self.params.regularize then self:regularize() end
    return err
end


----- Evaluate ----
function UniversalSchemaEncoder:evaluate()
    if self.params.test ~= '' then
        self:map(self.params.test, true)
    end
end


function UniversalSchemaEncoder:score_subdata(sub_data)
    local batches = {}
    self:gen_subdata_batches(sub_data, batches, 0, false)

    local scores = {}
    for i = 1, #batches do
        local ep_batch, rel_batch, _ = unpack(batches[i].data)
        if self.params.encoder == 'lookup-table' then rel_batch = rel_batch:view(rel_batch:size(1), 1) end
        if self.params.entEncoder == 'lookup-table' then ep_batch = ep_batch:view(ep_batch:size(1), 1) end
        local encoded_rel = self.rel_encoder(self:to_cuda(rel_batch)):squeeze()
        local encoded_ent = self.ent_encoder(self:to_cuda(ep_batch)):squeeze()
        local x = { encoded_rel, encoded_ent }
        local score = self.cosine(x):double()
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end

