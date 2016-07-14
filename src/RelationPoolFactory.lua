--
-- User: pat
-- Date: 3/11/16
--


local RelationPoolFactory = torch.class('RelationPoolFactory')


function RelationPoolFactory:add_random()
    local c = nn.ConcatTable_hacked():add(nn.Identity()):add(nn.RandomTensor(2, .25))
    return nn.Sequential():add(c):add(nn.CAddTable())
end


function RelationPoolFactory:pooling_layer(params, row_encoder, col_encoder, replicate)
    if params.relationPool == 'attention' then
        local weighting = nn.Sequential():add(nn.SoftMax())
--        local weighting = nn.Sequential():add(nn.Power(3)):add(nn.Normalize(1))
        local attention = self:weighted_relations(params, weighting, 3)
        return self:attention_layers(params, row_encoder, col_encoder, attention)

    elseif params.relationPool == 'log-sum-exp' then
        local log_sum_exp = nn.Sequential()
            :add(self:score_all_relations(1, 2, params.colDim, params.mlp))
                :add(nn.SplitTable(3))
                :add(nn.Sequencer(nn.LogSumExp(2)))
        return log_sum_exp -- self:attention_layers(params, row_encoder, col_encoder, log_sum_exp)

    elseif params.relationPool == 'topK' or params.relationPool == 'topk'then
        local weighting = nn.Sequential():add(nn.TopKSparse(params.k, 2)):add(nn.SoftMax())
        local topk =  self:weighted_relations(params, weighting, 3)
        return self:attention_layers(params, row_encoder, col_encoder, topk)

    elseif params.relationPool == 'max-relation' then
        local weighting = nn.Sequential():add(nn.MaxOneHot(2)):add(nn.SoftMax())
        local max_relation =  self:weighted_relations(params, weighting, 3)
        return self:attention_layers(params, row_encoder, col_encoder, max_relation)

    elseif params.relationPool == 'mean' then
        return nn.Sequential():add(nn.SelectTable_hacked(2)):add(col_encoder):add(nn.Mean(2)):add(replicate)

    elseif params.relationPool == 'max-pool' then
        return nn.Sequential():add(nn.SelectTable_hacked(2)):add(col_encoder):add(nn.Max(2)):add(replicate)

    else
        return nn.Sequential():add(nn.SelectTable_hacked(2)):add(col_encoder):add(replicate):add(nn.Squeeze(3))
    end
end

function RelationPoolFactory:attention_layers(params, row_encoder, col_encoder, weighting)
    local col_ouput = params.tieColViews and col_encoder:clone('weight', 'bias', 'gradWeight', 'gradBias') or col_encoder:clone()
    local pool = nn.Sequential()
        :add(nn.ConcatTable_hacked()
                :add(nn.SelectTable_hacked(1))
                :add(nn.SelectTable_hacked(2))
                :add(nn.SelectTable_hacked(2))
            )
        :add(nn.ParallelTable()
                :add(row_encoder) -- row
                :add(col_encoder) -- col input/attention
                :add(col_ouput) -- col output
            )
        :add(weighting)
    return pool
end

function RelationPoolFactory:weighted_relations(params, weighting, col_select_idx)
    local scoring_net = params.distanceType == 'cosine' and self:cosine_score_all_relations(1,2) or self:score_all_relations(1,2)
    -- input[1] = col-attention, input[2] = row, input[3] = col-output
    local relation_weight = nn.Sequential()
        -- score each relation with query
        :add(nn.ConcatTable_hacked()
            :add(nn.Sequential()
                :add(scoring_net)
                :add(nn.SplitTable(3))
                :add(nn.Sequencer(nn.Sequential()
                    :add(weighting)
                    :add(nn.Unsqueeze(3))
                    :add(nn.Transpose({2,3}))
                ))
                :add(nn.JoinTable(2))
            )
            :add(nn.Sequential()
                :add(nn.SelectTable(col_select_idx)) -- column output embeddings
            )
        )
        :add(nn.MM_fixed())
    return relation_weight
end

-- given a row and a set of columns, return the dot products between the row and each column
function RelationPoolFactory:score_all_relations(row_idx, col_idx)
    local row = nn.Sequential():add(nn.SelectTable(row_idx))
    local col = nn.Sequential():add(nn.SelectTable(col_idx)):add(nn.Transpose({2,3}))
    local relation_scorer = nn.Sequential()
        :add(nn.ConcatTable_hacked()
            :add(row)
            :add(col)
        )
        :add(nn.MM_fixed())
        :add(nn.Transpose({2,3}))
    return relation_scorer
end

-- given a row and a set of columns, return the cosine similarity between the row and each column
function RelationPoolFactory:cosine_score_all_relations(row_idx, col_idx)
    local row = nn.Sequential():add(nn.SelectTable(row_idx))
    local col = nn.Sequential():add(nn.SelectTable(col_idx))
    local dot_product = nn.Sequential()
        :add(nn.ConcatTable_hacked()
            :add(row)
            :add(nn.Sequential():add(col):add(nn.Transpose({2,3})))
        )
        :add(nn.MM_fixed())
        :add(nn.Transpose({2,3}))
    local norm = nn.Sequential():add(nn.Square()):add(nn.Sum(3)):add(nn.Sqrt())
    local norms = nn.Sequential()
        :add(nn.ConcatTable_hacked()
            :add(nn.Sequential():add(row:clone()):add(norm:clone()):add(nn.Unsqueeze(3)))
            :add(nn.Sequential():add(col:clone()):add(norm:clone()):add(nn.Unsqueeze(2)))
        )
        :add(nn.MM())
        :add(nn.AddConstant(1e-8))

    local relation_scorer = nn.Sequential()
        :add(nn.ConcatTable_hacked()
            :add(dot_product)
            :add(norms)
        )
        :add(nn.CDivTable())
    return relation_scorer
end