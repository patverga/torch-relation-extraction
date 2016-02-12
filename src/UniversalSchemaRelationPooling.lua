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


local UniversalSchemaRelationPool, parent = torch.class('UniversalSchemaRelationPool', 'UniversalSchemaEncoder')
local UniversalSchemaAttentionDot, parent = torch.class('UniversalSchemaAttentionDot', 'UniversalSchemaRelationPool')
local UniversalSchemaAttentionMatrix, parent = torch.class('UniversalSchemaAttentionMatrix', 'UniversalSchemaRelationPool')
local UniversalSchemaMax, parent = torch.class('UniversalSchemaMax', 'UniversalSchemaRelationPool')
local UniversalSchemaTopK, parent = torch.class('UniversalSchemaTopK', 'UniversalSchemaRelationPool')


local expand_as = function(input)
    local target_tensor = input[1]
    local orig_tensor = input[2]
    local expanded_tensor = torch.expand(orig_tensor, target_tensor:size())
    return expanded_tensor
end

local function make_attention(y_idx, hn_idx, dim)
    local term_1 = nn.Sequential():add(nn.SelectTable(y_idx)):add(nn.TemporalConvolution(dim, dim, 1))
    local term_2 = nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.SelectTable(y_idx))
            :add(nn.Sequential():add(nn.SelectTable(hn_idx)):add(nn.View(-1, 1, dim))))
        :add(grad.nn.AutoModule('AutoExpandAs')(expand_as)):add(nn.TemporalConvolution(dim, dim, 1))
    local concat = nn.ConcatTable():add(term_1):add(term_2)
    local M = nn.Sequential():add(concat):add(nn.CAddTable()):add(nn.Tanh())
    local alpha = nn.Sequential()
        :add(M):add(nn.TemporalConvolution(dim, 1, 1))
        :add(nn.SoftMax())
    local r = nn.Sequential():add(nn.ConcatTable():add(alpha):add(nn.SelectTable(y_idx)))
        :add(nn.MM(true)):add(nn.View(-1, dim))
    return r
end

-- given a row and a set of columns, return the maximum dot product between the row and any column
local function score_all_relations(row_idx, col_idx, dim)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.Sequential()
                :add(nn.ConcatTable()
                    :add(nn.SelectTable(col_idx))
                    :add(nn.Sequential():add(nn.SelectTable(row_idx)):add(nn.View(-1, 1, dim))))
                :add(grad.nn.AutoModule('AutoExpandAs')(expand_as)))
        :add(nn.SelectTable(col_idx)))
        :add(nn.CMulTable()):add(nn.Sum(3))
end

local top_K = function(input)
    local sorted, indices = torch.sort(input, 2, true)
    local k = 1
    local top = sorted:narrow(1,1,k)
    local sum = torch.sum(top, 2)
    local avg = torch.div(sum, k)
    return avg
end

function UniversalSchemaAttentionDot:build_scorer()
    local pos_score = nn.Sequential()
            :add(nn.ConcatTable()
            :add(make_attention(2, 1, self.params.colDim))
            :add(nn.SelectTable(1)))
            :add(nn.CMulTable()):add(nn.Sum(2))
    local neg_score = nn.Sequential()
            :add(nn.ConcatTable()
            :add(make_attention(2, 3, self.params.colDim))
            :add(nn.SelectTable(3)))
            :add(nn.CMulTable()):add(nn.Sum(2))

    local score_table = nn.ConcatTable()
        :add(pos_score):add(neg_score)
    return score_table
end

function UniversalSchemaAttentionMatrix:build_scorer()
    local pos_score = nn.Sequential():add(make_attention(2, 1, self.params.colDim)):add(nn.TemporalConvolution(self.params.colDim, 1, 1))
    local neg_score = nn.Sequential():add(make_attention(2, 3, self.params.colDim)):add(nn.TemporalConvolution(self.params.colDim, 1, 1))
    local score_table = nn.ConcatTable()
        :add(pos_score):add(neg_score)
    return score_table
end


function UniversalSchemaMax:build_scorer()
    local pos_score = score_all_relations(1, 2, self.params.colDim):add(nn.Max(2))
    local neg_score = score_all_relations(3, 2, self.params.colDim):add(nn.Max(2))
    local score_table = nn.ConcatTable()
        :add(pos_score):add(neg_score)
    return score_table
end

function UniversalSchemaTopK:build_scorer()
    local pos_score = score_all_relations(1, 2, self.params.colDim):add(grad.nn.AutoModule('AutoTopK')(top_K))
    local neg_score = score_all_relations(3, 2, self.params.colDim):add(grad.nn.AutoModule('AutoTopK')(top_K))
    local score_table = nn.ConcatTable()
        :add(pos_score):add(neg_score)
    return score_table
end



----- Evaluate ----

function UniversalSchemaRelationPool:score_subdata(sub_data)
    local batches = {}
    if sub_data.ep then self:gen_subdata_batches_four_col(sub_data, sub_data, batches, 0, false)
    else self:gen_subdata_batches_three_col(sub_data, sub_data, batches, 0, false) end

    local scores = {}
    for i = 1, #batches do
        local row_batch, col_batch, _ = unpack(batches[i].data)
        local encoded_row = self.row_encoder(row_batch):clone()
        local encoded_col = self.col_encoder(col_batch):clone()
        local x = {encoded_row, encoded_col}
        local score = self.net:get(2):get(1)(x):clone()
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end
