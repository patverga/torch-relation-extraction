--
-- User: pv
-- Date: 3/10/16
--
package.path = package.path .. ";src/?.lua;src/nn-modules/?.lua;src/eval/?.lua;src/classifier/?.lua;"
require 'AbstractSentenceScorer'


local CosineSentenceScorer, parent = torch.class('CosineSentenceScorer', 'AbstractSentenceScorer')
function CosineSentenceScorer:score_tac_relation(pattern_tensor, tac_tensor)
    if #self.text_encoder:findModules('nn.EncoderPool') > 0 then pattern_tensor = pattern_tensor:view(pattern_tensor:size(1), 1, pattern_tensor:size(2)) end
    if #self.kb_encoder:findModules('nn.EncoderPool') > 0 then tac_tensor = tac_tensor:view(tac_tensor:size(1), 1, tac_tensor:size(2)) end

    local tac_encoded = self.kb_encoder:forward(self:to_cuda(tac_tensor)):clone()
    local pattern_encoded = self.text_encoder:forward(self:to_cuda(pattern_tensor)):clone()

    if tac_encoded:dim() == 3 then tac_encoded = tac_encoded:view(tac_encoded:size(1), tac_encoded:size(3)) end
    if pattern_encoded:dim() == 3 then pattern_encoded = pattern_encoded:view(pattern_encoded:size(1), pattern_encoded:size(3)) end
    local x = { tac_encoded, pattern_encoded }

    local score = self:to_cuda(nn.CosineDistance())(x):double()
    return score
end

local NetworkScorer, _ = torch.class('NetworkScorer', 'AbstractSentenceScorer')
function NetworkScorer:score_tac_relation(pattern_tensor, tac_tensor)
    if #self.text_encoder:findModules('nn.EncoderPool') > 0 and pattern_tensor:dim() < 3 then
        pattern_tensor = pattern_tensor:view(pattern_tensor:size(1), 1, pattern_tensor:size(2))
    end
    local x = {tac_tensor:squeeze(), self:to_cuda(pattern_tensor), tac_tensor:clone()}
    local pattern_rel_scores = self.net:forward(x):clone()
    local scores = {}
    for i = 1, pattern_rel_scores:size(1) do
        table.insert(scores, pattern_rel_scores[i][1])
    end
    scores = torch.Tensor(scores)
    return scores
end


local SentenceClassifier, _ = torch.class('SentenceClassifier', 'AbstractSentenceScorer')
function SentenceClassifier:score_tac_relation(pattern_tensor, tac_tensor)
    if #self.text_encoder:findModules('nn.EncoderPool') > 0 then pattern_tensor = pattern_tensor:view(pattern_tensor:size(1), 1, pattern_tensor:size(2)) end
    local pattern_rel_scores = self.net:forward(self:to_cuda(pattern_tensor)):clone()
    local scores = {}
    for i = 1, pattern_rel_scores:size(1) do
        -- get score for the query relation
        table.insert(scores, pattern_rel_scores[i][tac_tensor[i][1]])
    end
    scores = torch.Tensor(scores)
    return scores
end


local PoolSentenceClassifier, _ = torch.class('PoolSentenceClassifier', 'AbstractSentenceScorer')
function PoolSentenceClassifier:score_tac_relation(pattern_tensor, tac_tensor)
    if #self.text_encoder:findModules('nn.EncoderPool') > 0 and pattern_tensor:dim() < 3 then
        pattern_tensor = pattern_tensor:view(pattern_tensor:size(1), 1, pattern_tensor:size(2))
    end

    local pattern_rel_scores = self.net:forward({tac_tensor, self:to_cuda(pattern_tensor)}):clone()

    local scores = {}
    for i = 1, pattern_rel_scores:size(1) do
        -- get score for the query relation
        table.insert(scores, pattern_rel_scores[i][tac_tensor[i][1]])
    end
    scores = torch.Tensor(scores)
    return scores
end

function PoolSentenceClassifier:run()
    -- process the candidate file
    local data = self:process_file(self:load_maps())
    -- seperate data by eps
    local ep_data = {}
    for _, sub_data in pairs(data) do
        if torch.type(sub_data) == 'table' then
            for i = 1, #sub_data.tac_tensor do
                local ep = sub_data.ep[i]
                local tac_idx = sub_data.tac_tensor[i][1][1]
                if ep_data[ep] == nil then ep_data[ep] = {} end
                if ep_data[ep][tac_idx] == nil then ep_data[ep][tac_idx] = {} end
                local padded_pattern = sub_data.pattern_tensor[i]
                if padded_pattern:size(2) < self.params.maxSeq then
                    local padding = padded_pattern:clone():resize(1,self.params.maxSeq-padded_pattern:size(2)):fill(self.params.padIdx)
                    padded_pattern = padded_pattern:cat(padding)
                end
                table.insert(ep_data[ep][tac_idx], padded_pattern:view(1,1,-1))
            end
        end
    end
    -- group data by number of relations for batching
    local pattern_count_data = {}
    for ep, tac_indices in pairs(ep_data) do
        for tac_idx, pattern_table in pairs(tac_indices) do
            local pattern_tensor = nn.JoinTable(2)(pattern_table)
            local count = pattern_tensor:size(2)
            if not pattern_count_data[count] then pattern_count_data[count] = {pattern_tensor={}, tac_tensor={}, ep={}} end
            table.insert(pattern_count_data[count].pattern_tensor, pattern_tensor)
            table.insert(pattern_count_data[count].tac_tensor, torch.Tensor(1,1):fill(tac_idx))
            table.insert(pattern_count_data[count].ep, ep)
        end
    end
    -- get a score for each ep,pattern
    local ep_tac_scores = {}
    for _, sub_data in pairs(pattern_count_data) do
        local pattern_tensor = nn.JoinTable(1)(sub_data.pattern_tensor)
        local tac_tensor = nn.JoinTable(1)(sub_data.tac_tensor)
        local scores = self:score_tac_relation(pattern_tensor, tac_tensor)
        for i = 1, scores:size(1) do
            if ep_tac_scores[sub_data.ep[i]] == nil then ep_tac_scores[sub_data.ep[i]] = {} end
            ep_tac_scores[sub_data.ep[i]][tac_tensor[i][1]] = scores[i]
        end
    end
    -- score and export candidate file
    local max_scores, max_score, min_score, out_lines = self:score_data(data, ep_tac_scores)
    self:write_output(max_scores, max_score, min_score, out_lines)
    print ('\nDone, found ' .. self.in_vocab .. ' in vocab tokens and ' .. self.out_vocab .. ' out of vocab tokens.')
end