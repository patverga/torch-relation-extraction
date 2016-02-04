--
-- User: pat
-- Date: 1/29/16
--

package.path = package.path .. ";src/?.lua"

require 'torch'
require 'rnn'
require 'optim'
require 'UniversalSchemaEncoder'
require 'nn-modules/ReplicateAs'
require 'nn-modules/ViewTable'
require 'nn-modules/VariableLengthConcatTable'

grad = require 'autograd'
grad.optimize(true) -- global



local UniversalSchemaAttention, parent = torch.class('UniversalSchemaAttention', 'UniversalSchemaEncoder')


local auto_term_2 = function(input)
    local cols = input[1]
    local row = input[2]
    local row_matrix = torch.expand(row, cols:size())
    return row_matrix
end

local function make_attention(y_idx, hn_idx, dim)
    local term_1 = nn.Sequential():add(nn.SelectTable(y_idx)):add(nn.TemporalConvolution(dim, dim, 1))
    local term_2 = nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.SelectTable(y_idx))
            :add(nn.Sequential():add(nn.SelectTable(hn_idx)):add(nn.View(-1, 1, dim))))
        :add(grad.nn.AutoModule('AutoTerm2')(auto_term_2)):add(nn.TemporalConvolution(dim, dim, 1))
    local concat = nn.ConcatTable():add(term_1):add(term_2)
    local M = nn.Sequential():add(concat):add(nn.CAddTable()):add(nn.Tanh())
    local alpha = nn.Sequential()
        :add(M):add(nn.TemporalConvolution(dim, 1, 1))
        :add(nn.SoftMax())
    local r = nn.Sequential():add(nn.ConcatTable():add(alpha):add(nn.SelectTable(y_idx)))
        :add(nn.MM(true)):add(nn.View(-1, dim))
    return r
end

function UniversalSchemaAttention:build_network(pos_row_encoder, col_encoder)
    local neg_row_encoder = pos_row_encoder:clone()

    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
    loading_par_table:add(pos_row_encoder)
    loading_par_table:add(col_encoder)
    loading_par_table:add(neg_row_encoder)


    local pos_concat = nn.ConcatTable()
        :add(make_attention(2, 1, self.params.colDim))
        :add(nn.SelectTable(1))
    local pos_dot = nn.Sequential():add(pos_concat):add(nn.CMulTable()):add(nn.Sum(2))

    local neg_concat = nn.ConcatTable()
        :add(make_attention(2, 3, self.params.colDim))
        :add(nn.SelectTable(3))
    local neg_dot = nn.Sequential():add(neg_concat):add(nn.CMulTable()):add(nn.Sum(2))

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


----- Evaluate ----

--function UniversalSchemaAttention:score_subdata(sub_data)
--    local batches = {}
--    if sub_data.ep then self:gen_subdata_batches_four_col(sub_data, sub_data, batches, 0, false)
--    else self:gen_subdata_batches_three_col(sub_data, sub_data, batches, 0, false) end
--
--    local scores = {}
--    for i = 1, #batches do
--        local row_batch, col_batch, _ = unpack(batches[i].data)
--        if self.params.colEncoder == 'lookup-table' then col_batch = col_batch:view(col_batch:size(1), 1) end
--        if self.params.rowEncoder == 'lookup-table' then row_batch = row_batch:view(row_batch:size(1), 1) end
--        local encoded_rel = self.col_encoder(self:to_cuda(col_batch)):squeeze():clone()
--        local encoded_ent = self.row_encoder(self:to_cuda(row_batch)):squeeze()
--        local x = { encoded_rel, encoded_ent }
--        local score = self.cosine(x):double()
--        table.insert(scores, score)
--    end
--
--    return scores, sub_data.label:view(sub_data.label:size(1))
--end

