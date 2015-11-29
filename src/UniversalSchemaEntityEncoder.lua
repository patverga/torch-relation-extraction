--
-- User: pat
-- Date: 9/21/15
--

require 'torch'
require 'rnn'
require 'optim'
require 'RelationEncoderModel.lua'

local UniversalSchemaEntityEncoder, parent = torch.class('UniversalSchemaEntityEncoder', 'RelationEncoderModel')

function UniversalSchemaEntityEncoder:__init(params, rel_table, encoder, squeeze_rel)
    self.__index = self
    self.params = params
    self:init_opt()
    self.squeeze_rel = squeeze_rel or false
    self.train_data = self:load_entity_data(params.train)

    -- cosine distance network for evaluation
    self.cosine = nn.CosineDistance()
    self:to_cuda(self.cosine)

    -- either load model from file or initialize new one
    if params.loadModel ~= '' then
        local loaded_model = torch.load(params.loadModel)
        self.net = self:to_cuda(loaded_model.net)
        encoder = self.net:get(1):get(3) --self:to_cuda(loaded_model.encoder)
        self.ent_table = self.net:get(1):get(1) --self:to_cuda((loaded_model.ent_table or self.net:get(1):get(1)))
        rel_table = self:to_cuda(loaded_model.rel_table)
        self.opt_state = loaded_model.opt_state
    else
        self.net, self.ent_table, self.ep_encoder = self:build_network(params, self.train_data.num_ents, encoder)
    end
    self.rel_table = rel_table
    self.encoder = encoder
end


function UniversalSchemaEntityEncoder:build_network(params, num_ents, encoder)
    -- seperate lookup tables for entity pairs and relations
    local pos_e1_table = nn.LookupTable(num_ents, params.embeddingDim)
    -- preload entity pairs
    if params.loadEpEmbeddings ~= '' then pos_e1_table.weight = self:to_cuda(torch.load(params.loadEpEmbeddings))
    else pos_e1_table.weight = torch.rand(num_ents, params.embeddingDim):add(-.5):mul(0.1)
    end

    local pos_e2_table = pos_e1_table:clone()
    local neg_e1_table = pos_e1_table:clone()
    local neg_e2_table = pos_e1_table:clone()

    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_e1_table)
    loading_par_table:add(pos_e2_table)
    loading_par_table:add(encoder)
    loading_par_table:add(neg_e1_table)
    loading_par_table:add(neg_e2_table)

    local ep_encoder = nn.Sequential()
    ep_encoder:add(nn.JoinTable(2))
    if params.compositional then
        ep_encoder:add(nn.Linear(params.embeddingDim*2, params.embeddingDim))
        ep_encoder:add(nn.ReLU())
        ep_encoder:add(nn.Linear(params.embeddingDim, params.embeddingDim))
    end

    -- layers to compute the dot prduct of the positive and negative samples
    local pos_ents = nn.Sequential()
    pos_ents:add(nn.NarrowTable(1, 2))
    pos_ents:add(ep_encoder)

    local pos_concat = nn.ConcatTable()
    pos_concat:add(pos_ents)
    pos_concat:add(nn.SelectTable(3))
    local pos_dot = nn.Sequential()
    pos_dot:add(pos_concat)
    pos_dot:add(nn.CMulTable())
    pos_dot:add(nn.Sum(2))

    local neg_ents = nn.Sequential()
    neg_ents:add(nn.NarrowTable(4, 2))
    neg_ents:add(ep_encoder:clone())

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
    neg_e1_table:share(neg_e2_table, 'weight', 'bias', 'gradWeight', 'gradBias')
    pos_e2_table:share(neg_e1_table, 'weight', 'bias', 'gradWeight', 'gradBias')
    pos_e1_table:share(pos_e2_table, 'weight', 'bias', 'gradWeight', 'gradBias')


    return net, pos_e1_table, ep_encoder
end



----- TRAIN -----
function UniversalSchemaEntityEncoder:gen_subdata_batches(sub_data, batches, max_neg, shuffle)
    local start = 1
    local rand_order = shuffle and torch.randperm(sub_data.ep:size(1)):long() or torch.range(1, sub_data.ep:size(1)):long()
    while start <= sub_data.ep:size(1) do
        local size = math.min(self.params.batchSize, sub_data.ep:size(1) - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local pos_e1_batch = self:to_cuda(sub_data.e1:index(1, batch_indices))
        local pos_e2_batch = self:to_cuda(sub_data.e2:index(1, batch_indices))
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


        local rel_batch = self:to_cuda(self.params.testing and sub_data.rel:index(1, batch_indices) or sub_data.seq:index(1, batch_indices))
        if self.squeeze_rel then rel_batch = rel_batch:squeeze() end
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
    self.rel_table.weight:renorm(2, 2, 3.0)
    self.ent_table.weight:renorm(2, 2, 3.0)
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



--- - Evaluate ----
function UniversalSchemaEntityEncoder:evaluate()
    if self.params.test ~= '' then
        self:map(self.params.test, true)
    end
end

function UniversalSchemaEntityEncoder:score_subdata(sub_data)
    local batches = {}
    self:gen_subdata_batches(sub_data, batches, 0, false)

    local scores = {}
    for i = 1, #batches do
        local e1_batch, e2_batch, rel_batch, _, _ = unpack(batches[i].data)
        if self.params.testing then rel_batch = rel_batch:contiguous():view(rel_batch:size(1), 1) end
        local encoded_rel = self.encoder:forward(self:to_cuda(rel_batch))
        if encoded_rel:dim() == 3 then encoded_rel = encoded_rel:view(encoded_rel:size(1), encoded_rel:size(3)) end
        local e1 = self.ent_table(self:to_cuda(e1_batch:contiguous())):clone()
        e1 = e1:view(e1:size(1),e1:size(3))
        local e2 = self.ent_table(self:to_cuda(e2_batch:contiguous())):clone()
        e2 = e2:view(e2:size(1), e2:size(3))
        local ep = self.ep_encoder({e1, e2})
        if ep:dim() == 3 then ep = ep:view(ep:size(1), ep:size(3)) end
        local x = { ep, encoded_rel, }
        local score = self.cosine(x):double()
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end

