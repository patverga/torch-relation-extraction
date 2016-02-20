--
-- User: pat
-- Date: 2/17/16
--

local PositiveOnlyUniversalSchema, parent = torch.class('PositiveOnlyUniversalSchema', 'UniversalSchemaEncoder')


function PositiveOnlyUniversalSchema:build_scorer()
    local pos_score = nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.SelectTable(3)) -- tensor of ones
            :add(nn.Sequential() -- exp(-theta)
                :add(nn.NarrowTable(1, 2))
                :add(nn.CMulTable())
                :add(nn.Sum(2))
                :add(nn.MulConstant(-1))
                :add(nn.Exp())))
        :add(nn.CSubTable())

    return pos_score
end

function PositiveOnlyUniversalSchema:build_network(pos_row_encoder, col_encoder)
    -- load the eps and rel
    local loading_par_table = nn.ParallelTable()
        :add(pos_row_encoder)
        :add(col_encoder)
        :add(nn.Identity())
    local net = nn.Sequential():add(loading_par_table):add(self:build_scorer())
    self:to_cuda(net)
    return net
end


function PositiveOnlyUniversalSchema:regularize_hooks()
--    self.col_table.weight:clamp(0, 1)
--    self.row_table.weight:clamp(0, 1)
end


function PositiveOnlyUniversalSchema:gen_subdata_batches_three_col(data, sub_data, batches, max_neg, shuffle)
    local start = 1
    local rand_order = shuffle and torch.randperm(sub_data.row:size(1)):long() or torch.range(1, sub_data.row:size(1)):long()
    while start <= sub_data.row:size(1) do
        local size = math.min(self.params.batchSize, sub_data.row:size(1) - start + 1)
        local batch_indices = rand_order:narrow(1, start, size)
        local pos_row_batch = self.params.rowEncoder == 'lookup-table' and sub_data.row:index(1, batch_indices) or sub_data.row_seq:index(1, batch_indices)
        local col_batch = self.params.colEncoder == 'lookup-table' and sub_data.col:index(1, batch_indices) or sub_data.col_seq:index(1, batch_indices)
        local pos_batch = {pos_row_batch, col_batch, self:to_cuda(torch.ones(size)) }
        table.insert(batches, { data = pos_batch, label = self:to_cuda(torch.ones(size)) })

        -- add negatives
        local neg_row_batch = self:gen_neg(data, pos_row_batch, size, max_neg)
        local neg_batch = {neg_row_batch, col_batch:clone(), self:to_cuda(torch.ones(size)) }
        table.insert(batches, { data = neg_batch, label = self:to_cuda(torch.zeros(size)) })
        start = start + size
    end
end


function PositiveOnlyUniversalSchema:score_subdata(sub_data)
    local batches = {}
    if sub_data.ep then print('Must supply 3 col data'); os.exit()
    else self:gen_subdata_batches_three_col(sub_data, sub_data, batches, 0, false) end

    local scores = {}
    for i = 1, #batches do
        local row_batch, col_batch, ones = unpack(batches[i].data)
        local score = self.net({row_batch:squeeze(), col_batch:squeeze(), ones})
        table.insert(scores, score)
    end

    return scores, sub_data.label:view(sub_data.label:size(1))
end
