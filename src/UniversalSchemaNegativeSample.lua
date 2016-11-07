--
-- User: pv
-- Date: 3/10/16
--

require 'RelationPoolFactory'
local UniversalSchemaNegativeSample, parent = torch.class('UniversalSchemaNegativeSample', 'UniversalSchemaEncoder')


function UniversalSchemaNegativeSample:__init(params, row_table, row_encoder, col_table, col_encoder, use_entities)
    if params.loadModel == '' and params.rowEncoder ~= 'lookup-table' and #row_encoder:findModules('nn.EncoderPool') == 0 then row_encoder = nn.EncoderPool(row_encoder:clearState()) end
    parent.__init(self, params, row_table, row_encoder, col_table, col_encoder, use_entities)
    self.reshape = self.reshape or self.net:findModules('nn.Reshape')[1]
    self.replicate = self.replicate or self.net:findModules('nn.Replicate')[1]
    -- backwards compatible for stupid updates in rnn
    for _, s in pairs(self.net:findModules('nn.Sequencer')) do
        s.tableoutput, s.tablegradInput = {}, {}
    end

end


function UniversalSchemaNegativeSample:build_network(row_encoder, col_encoder)

    local row_attention_encoder = row_encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')

    -- store this so we can perform evaluation more efficiently
    self.replicate = nn.Replicate(self.params.negSamples+1, 2)
    self.reshape = nn.Reshape(self.params.negSamples, -1)

    local cols = nn.Sequential():add(nn.SelectTable(2))
    if (self.params.patternDropout > 0) then cols:add(nn.PatternDropout(self.params.patternDropout)) end

    -- merge the positive and negative rows and concat with cols
    local all_rows_and_cols = nn.Sequential()
        :add(nn.ConcatTable_hacked()
            :add(nn.Sequential() -- merge positive and negative rows
                :add(nn.ConcatTable_hacked()
                    :add(nn.Sequential()
                        :add(nn.SelectTable(1))
                        :add(nn.Unsqueeze_fixed(2))
                    )
                    :add(nn.Sequential()
                        :add(nn.SelectTable(3))
                        :add(self.reshape)
                    )
                )
                :add(nn.JoinTable_hacked(2))
            )
            :add(cols)
        )

    local net = nn.Sequential()
        :add(all_rows_and_cols)
        :add(nn.ConcatTable_hacked()
            :add(nn.Sequential()
                :add(RelationPoolFactory:pooling_layer(self.params, row_attention_encoder, col_encoder, self.replicate))
            )
            :add(nn.Sequential()
                :add(nn.SelectTable(1))
                :add(row_encoder)
            )
        )
        :add(self:build_scorer())
    self:to_cuda(net)
    return net
end




function UniversalSchemaNegativeSample:build_scorer()
    local scorer
    if self.params.distanceType == 'cosine' then
        scorer = nn.Sequential():add(nn.CosineDistance(nn.ConcatTable():add(nn.Sequential():add(nn.SelectTable(1)):add(nn.SplitTable(2)):add(nn.SelectTable(1))):add(nn.Sequential():add(nn.SelectTable(2)):add(nn.SplitTable(2)):add(nn.SelectTable(1))))):add(nn.Unsqueeze(2))
   else
        scorer = nn.Sequential()
            :add(nn.CMulTable())
            :add(nn.Sum(3))
    end
    if self.params.criterion == 'bce' then scorer:add(nn.Sigmoid()) end
    if self.params.criterion == 'bpr' then scorer:add(nn.SplitTable(2)) end
    return scorer
end


function UniversalSchemaNegativeSample:score_subdata(sub_data)
    local batches = {}
    if sub_data.ep then self:gen_subdata_batches_four_col(sub_data, sub_data, batches, 0, false)
    else self:gen_subdata_batches_three_col(sub_data, sub_data, batches, 0, false) end
    local scores = {}

    -- set replication to 2 for quicker evaluation
    if self.replicate then self.replicate.nfeatures = 2 end
    if self.reshape then self.reshape.size[1] = 1; self.reshape.nelement = -1; self.reshape.batchsize[2] = 1 end
    for i = 1, #batches do
        local row_batch, col_batch, neg_batch = unpack(batches[i].data)
        if self.params.rowEncoder == 'lookup-table' then row_batch = row_batch:view(row_batch:size(1)) else row_batch = row_batch:view(row_batch:size(1),1) end
        if self.params.colEncoder == 'lookup-table'  then col_batch = col_batch:view(col_batch:size(1), col_batch:size(2)) end
        local x = { self:to_cuda(row_batch), self:to_cuda(col_batch), self:to_cuda(neg_batch) }
        local score = self.net(x)
        score = self.params.criterion == 'bpr' and score[1]:double() or score:double():select(2,1) -- first column is positive scores
        table.insert(scores, score)
    end
    -- set replication back where we want it
    if self.replicate then self.replicate.nfeatures = self.params.negSamples+1 end
    if self.reshape then self.reshape.size[1] = self.params.negSamples; self.reshape.nelement = -self.params.negSamples; self.reshape.batchsize[2] = self.params.negSamples end
    return scores, sub_data.label:view(sub_data.label:size(1))
end


function UniversalSchemaNegativeSample:tac_eval(model_file, out_dir, eval_args)
    self.reshape.size[1] = 1; self.reshape.nelement = -1; self.reshape.batchsize[2] = 1
    parent.tac_eval(self, model_file, out_dir, eval_args)
    self.reshape.size[1] = self.params.negSamples;
    self.reshape.nelement = -self.params.negSamples;
    self.reshape.batchsize[2] = self.params.negSamples
end


function UniversalSchemaNegativeSample:accuracy(file)
    local total_correct = 0.0
    local total_count = 0.0
    local data = torch.load(file)
    -- 41 tac relations
    self.replicate = self.replicate or  self.net:get(1):get(2):get(2)
    self.replicate.nfeatures = 41
    local old_neg_samples= self.params.negSamples
    self.params.negSamples = 1
    for _, sub_data in pairs(data) do
        if torch.type(sub_data) == 'table' then
            -- score each of the test samples
            local batches = {}
            if sub_data.ep then self:gen_subdata_batches_four_col(sub_data, sub_data, batches, 0, false)
            else self:gen_subdata_batches_three_col(sub_data, sub_data, batches, 0, false) end
            for i = 1, #batches do
                local row_batch, col_batch, neg_batch = unpack(batches[i].data)
                -- generate negatives for all tac relations except the correct one
                neg_batch = torch.range(1,41):view(1,-1):expandAs(torch.Tensor(col_batch:size(1), 41))
                neg_batch = neg_batch:maskedSelect(neg_batch:ne(row_batch:expandAs(neg_batch))):view(row_batch:size(1), 40)
                local x = {row_batch:squeeze(), col_batch, neg_batch }
                total_count = total_count + col_batch:size(1)
                local pattern_rel_scores = self.net(x)--:clone()
                local _, max_indices = torch.max(pattern_rel_scores, 2)
                total_correct = total_correct + max_indices:eq(self:to_cuda(batches[i].label)):sum()
            end
        end
    end
    self.params.negSamples = old_neg_samples
    self.replicate.nfeatures = self.params.negSamples+1
    local accuracy = (total_correct / total_count) * 100
    print('Accuracy : ' .. accuracy)
    return accuracy
end


