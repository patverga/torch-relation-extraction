--
-- User: pat
-- Date: 8/3/15
--

local PatternScorer = torch.class('PatternScorer')

--- tac relations
local test_relations = {
    'org:alternate_names',
    'org:city_of_headquarters',
    'org:country_of_headquarters',
    'org:date_founded',
    'org:founded_by',
    'org:member_of',
    'org:members',
    'org:number_of_employees_members',
    'org:parents',
    'org:political_religious_affiliation',
    'org:shareholders',
    'org:stateorprovince_of_headquarters',
    'org:subsidiaries',
    'org:top_members_employees',
    'org:website',
    'per:age',
    'per:alternate_names',
    'per:charges',
    'per:children',
    'per:cities_of_residence',
    'per:city_of_birth',
    'per:countries_of_residence',
    'per:country_of_birth',
    'per:date_of_birth',
    'per:date_of_death',
    'per:employee_or_member_of',
    'per:origin',
    'per:other_family',
    'per:parents',
    'per:religion',
    'per:schools_attended',
    'per:siblings',
    'per:spouse',
    'per:stateorprovince_of_death',
    'per:statesorprovinces_of_residence',
    'per:title',
}

function PatternScorer:new(rel_map, rel_weights, high_score, o)
    o = o or {}
    setmetatable(o, self)
    self.__index = self
    self.rel_weights = rel_weights
    self.rel_map = rel_map
    self.reverse_rel_map = {}
    if rel_map then
        for s, i in pairs(rel_map) do
            self.reverse_rel_map[i] = s
        end
    end
    -- higher score is better, else lower score is better
    self.high_score = high_score or true
end


local function get_top_patterns(tr_index, rel_weights, high_score)
    local tr_mag = torch.sqrt(torch.sum(torch.pow(rel_weights[tr_index], 2)))
    -- cos distance
    local rel_mag = torch.sqrt(torch.sum(torch.pow(rel_weights, 2), 2))
    local similarities = torch.cdiv(rel_weights * rel_weights[tr_index], rel_mag * tr_mag)
    -- sort in descending order
    local y, j = torch.sort(similarities, high_score)
    return y, j
end


--- for a given relation, find all relations that have a distance < threshold
function PatternScorer:get_top_patterns_threshold(threshold)
    for i = 1, #test_relations do
        print('patterns with similarity > ' .. threshold .. ' for relation : ' .. test_relations[i])
        local scores, indices = get_top_patterns(self.rel_map[test_relations[i]], self.rel_weights, self.high_score)
        for j = 1, indices:size(1) do
            if (scores[j] < threshold) then break end
            print(scores[j], self.reverse_rel_map[indices[j]])
        end
    end
end

-- for a given relation, find the k most similar other relations
function PatternScorer:get_top_patterns_topk(k)
    for i = 1, #test_relations do
        print('Top ' .. k .. ' patterns for relation : ' .. test_relations[i])
        local scores, indices = get_top_patterns(self.rel_map[test_relations[i]], self.rel_weights, self.high_score)
        for j = 1, k do
            print(scores[j], self.reverse_rel_map[indices[j]])
        end
    end
end



