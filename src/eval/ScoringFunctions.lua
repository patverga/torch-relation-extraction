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
        pattern_tensor = pattern_tensor:view(-1, 1, pattern_tensor:size(2))
    end
    local reshape = self.net:findModules('nn.Reshape')[1]
    if reshape then reshape.size[1] = 1; reshape.nelement = -1; reshape.batchsize[2] = 1 end
    tac_tensor = self:to_cuda(tac_tensor)
    local x = {tac_tensor, self:to_cuda(pattern_tensor), tac_tensor }
    local pattern_rel_scores = self.net:forward(x)
    pattern_rel_scores = torch.type(pattern_rel_scores) == 'table' and pattern_rel_scores[1]:clone():view(-1,1) or pattern_rel_scores:clone()
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
