--
-- User: pat
-- Date: 8/26/15
--

require 'torch'
require 'rnn'
require 'optim'
require 'UniversalSchemaEncoder.lua'

local UniversalSchemaJointEncoder, parent = torch.class('UniversalSchemaJointEncoder', 'UniversalSchemaEncoder')

function UniversalSchemaJointEncoder:__init(params, kb_rel_table, text_encoder, squeeze_rel)
    self.__index = self
    self.params = params
    self:init_opt()
    self.squeeze_rel = squeeze_rel or false
    self.train_data = self:load_ep_data(params.train)

    -- cosine distance network for evaluation
    self.cosine = nn.CosineDistance()
    self:to_cuda(self.cosine)

-- either load model from file or initialize new one
    if params.loadModel ~= '' then
        local loaded_model = torch.load(params.loadModel)
        self.kb_net = self:to_cuda(loaded_model.kb_net)
        self.text_net = self:to_cuda(loaded_model.text_net)
        text_encoder = self:to_cuda(loaded_model.text_encoder)
        self.ent_table = self:to_cuda((loaded_model.ent_table or self.net:get(1):get(1)))
        kb_rel_table = self:to_cuda(loaded_model.kb_rel_table)
        self.kb_state = loaded_model.kb_state
        self.text_state = loaded_model.text_state
    else
        -- seperate lookup tables for entity pairs and relations
        local pos_ep_table = self:to_cuda(nn.LookupTable(self.train_data.num_eps, params.embeddingDim))
        -- preload entity pairs
        if params.loadEpEmbeddings ~= '' then pos_ep_table.weight = (self:to_cuda(torch.load(params.loadEpEmbeddings)))
        else pos_ep_table.weight = pos_ep_table.weight:normal(0, 1):mul(1 / params.embeddingDim)
        end
        local neg_ep_table = pos_ep_table:clone()
        self.ent_table = pos_ep_table

        -- kb_network
        self.kb_net = self:build_net(pos_ep_table, kb_rel_table, neg_ep_table)
        print ('KB-Net: ', self.kb_net)
        -- text network
        self.text_net = self:build_net(pos_ep_table, text_encoder, neg_ep_table)
        print ('Text-Net: ', self.text_net)

        pos_ep_table:share(neg_ep_table, 'weight', 'bias', 'gradWeight', 'gradBias')

        self.kb_state = {}
        self.text_state = {}

    end
    self.kb_rel_table = kb_rel_table
    self.text_encoder = text_encoder

end

function UniversalSchemaJointEncoder:build_net(pos_ep_table, encoder, neg_ep_table)
    local par_loading_table = nn.ParallelTable()
    par_loading_table:add(pos_ep_table)
    par_loading_table:add(encoder)
    par_loading_table:add(neg_ep_table)

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
    local concat_table = nn.ConcatTable()
    concat_table:add(pos_dot)
    concat_table:add(neg_dot)

    local net = nn.Sequential()
    net:add(par_loading_table)
    net:add(concat_table)
    self:to_cuda(net)

    return net
end



----- TRAIN -----

function UniversalSchemaJointEncoder:train(num_epochs)
    num_epochs = num_epochs or self.params.numEpochs
    -- optim stuff
    local kb_parameters, kb_gradParameters = self.kb_net:getParameters()
    local text_parameters, text_gradParameters = self.text_net:getParameters()

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
            local net, parameters, gradParameters, opt_state
            if batch.data[2]:size(2) == 1 then
                net = self.kb_net
                parameters = kb_parameters
                gradParameters = kb_gradParameters
                opt_state = self.kb_state
                batch.data[2] = batch.data[2]:squeeze()
            else
                net = self.text_net
                parameters = text_parameters
                gradParameters = text_gradParameters
                opt_state = self.text_state
            end
            epoch_error = epoch_error + self:optim_update(net, self.crit, example, label, parameters, gradParameters, self.opt_config, opt_state, epoch)

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



--- - Evaluate ----

function UniversalSchemaJointEncoder:score_subdata(sub_data)
    local batches = {}
    self:gen_subdata_batches(sub_data, batches, 0, false)

    local scores = {}
    for i = 1, #batches do
        local ep_batch, rel_batch, _ = unpack(batches[i].data)
        if self.params.testing then rel_batch = rel_batch:contiguous():view(rel_batch:size(1), 1) end
        local encoded_rel = self.kb_rel_table:forward(self:to_cuda(rel_batch))
--        local encoded_rel = self.text_encoder:forward(self:to_cuda(rel_batch))
        local x = { encoded_rel, self.ent_table(self:to_cuda(ep_batch:contiguous():view(ep_batch:size(1), 1))) }
        x = {
            x[1]:view(x[2]:size(1), x[2]:size(3)),
            x[2]:view(x[2]:size(1), x[2]:size(3))
        }
        local score = self.cosine(x):double()
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end


--- IO ---
function UniversalSchemaJointEncoder:save_model(epoch)
    if self.params.saveModel ~= '' then
        torch.save(self.params.saveModel .. '/' .. epoch .. '-model',
            {kb_net = self.kb_net, text_net = self.text_net, text_encoder = self.text_encoder, kb_rel_table = self.kb_rel_table, ent_table = self.ent_table, opt_state = self.opt_state})
        self:tac_eval(self.params.saveModel .. '/' .. epoch, self.params.otherArgs)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-ent-weights', self.params.gpuid >= 0 and self.ent_table.weight:double() or self.ent_table.weight)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-token-weights', self.params.gpuid >= 0 and self.text_encoder:get(1).weight:double() or self.text_encoder:get(1).weight)
        torch.save(self.params.saveModel .. '/' .. epoch .. '-kb-weights', self.params.gpuid >= 0 and self.kb_rel_table.weight:double() or self.kb_rel_table.weight)
    end
end