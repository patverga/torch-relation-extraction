--
-- User: pat
-- Date: 9/21/15
--

require 'torch'
require 'rnn'
require 'optim'
require 'UniversalSchemaEncoder'

local UniversalSchemaEntityEncoder, parent = torch.class('UniversalSchemaEntityEncoder', 'UniversalSchemaEncoder')

function UniversalSchemaEntityEncoder:build_network(pos_row_encoder, col_encoder)
    local neg_row_encoder = pos_row_encoder:clone()

    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_row_encoder)
    loading_par_table:add(col_encoder)
    loading_par_table:add(neg_row_encoder)

    local rel = nn.SelectTable(2)

    local pos_e1 = nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(1))
    local pos_e2 = nn.Sequential():add(nn.SelectTable(1)):add(nn.SelectTable(2))
    local pos_ep = nn.Sequential():add(nn.ConcatTable():add(pos_e1):add(pos_e2)):add(nn.JoinTable(2))
    local pos_ep_rel = nn.ConcatTable():add(rel:clone()):add(pos_ep)

    local neg_e1 = nn.Sequential():add(nn.SelectTable(3)):add(nn.SelectTable(1))
    local neg_e2 = nn.Sequential():add(nn.SelectTable(3)):add(nn.SelectTable(2))
    local neg_ep = nn.Sequential():add(nn.ConcatTable():add(neg_e1):add(neg_e2)):add(nn.JoinTable(2))
    local neg_ep_rel = nn.ConcatTable():add(rel:clone()):add(neg_ep)

    -- layers to compute the dot prduct of the positive and negative samples
    local pos_dot = nn.Sequential()
    pos_dot:add(pos_ep_rel)
    pos_dot:add(nn.CMulTable())
    pos_dot:add(nn.Sum(2))

    local neg_dot = nn.Sequential()
    neg_dot:add(neg_ep_rel)
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

function UniversalSchemaEntityEncoder:score_subdata(sub_data)
    local batches = {}
    if sub_data.ep then self:gen_subdata_batches_four_col(sub_data, sub_data, batches, 0, false)
    else self:gen_subdata_batches_three_col(sub_data, sub_data, batches, 0, false) end

    local scores = {}
    for i = 1, #batches do
        local row_batch, col_batch, _ = unpack(batches[i].data)
        if self.params.colEncoder == 'lookup-table' then col_batch = col_batch:view(col_batch:size(1), 1) end
        if self.params.rowEncoder == 'lookup-table' then row_batch = row_batch:view(row_batch:size(1), 1) end
        local encoded_rel = self.col_encoder(self:to_cuda(col_batch)):squeeze():clone()
        local encoded_ep = self:to_cuda(nn.JoinTable(2))(self.row_encoder(self:to_cuda(row_batch)))
        local x = { encoded_rel, encoded_ep }
        local score = self.cosine(x):double()
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end

