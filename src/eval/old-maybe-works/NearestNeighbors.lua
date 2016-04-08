package.path = package.path .. ";src/?.lua"

require 'rnn'
require 'nn-modules/ViewTable'

local cmd = torch.CmdLine()
cmd:option('-map', '', 'txt file containing vocab-index map')
cmd:option('-embeddings', '', 'learned embeddings')
cmd:option('-model', '', 'trained model with a text encoder [optional, uses embeddings directly otherwise]')
cmd:option('-topK', 5, 'number of top nearest nieghbors to find')
cmd:option('-input', '', 'input token to find nearest neighbors for')
cmd:option('-inputFile', '', 'file of inputs to query')
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')
cmd:option('-delim', ' ', 'split tokens on this')
cmd:option('-esOnly', false, 'only return tokens ending in @es')
cmd:option('-enOnly', false, 'only return tokens NOT ending in @es')
cmd:option('-bothLang', false, 'return english and spanish top results seperately')
cmd:option('-tacOnly', false, 'only return tac relations')
cmd:option('-dictionary', '', 'check if tokens are in translation dictionary')
cmd:option('-relations', false, 'use full relations instead of tokens')

local params = cmd:parse(arg)
local function to_cuda(x) return params.gpuid >= 0 and x:cuda() or x end
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local function load_map(map_file)
    local token_count = 0
    local string_idx_map = {}
    local idx_string_map = {}
    for line in io.lines(map_file) do
        local token, idx = string.match(line, "([^\t]+)\t([^\t]+)")
        if token ~= nil and idx ~= nil then
            token_count = token_count + 1
            string_idx_map[token] = tonumber(idx)
            idx_string_map[tonumber(idx)] = token
        end
    end
    return token_count,  string_idx_map, idx_string_map
end

local function load_dictionary_file(file, delim)
    local dictionary = {}
    if file ~= '' then
        for line in io.lines(file) do
            local en, es = string.match(line, "([^" .. delim .. "]+)" .. delim .. "([^" .. delim .. "]+)")
            if en ~= nil and es ~= nil then dictionary[en] = es  end
        end
    end
    return dictionary
end


local function load_input_file(file, delim)
    local inputs = {}
    if file ~= '' then
        for line in io.lines(file) do
            local input, _ = string.match(line, "([^" .. delim .. "]+)" .. delim .. "([^" .. delim .. "]+)")
            if input then table.insert(inputs, input) end
        end
    end
    return inputs
end

local function nearest_neighbor(idx, data)
    local cos = nn.CosineDistance()
    local max_sim = 0
    local max_idx = 0
    for i = 1, data:size(1) do
        if (i ~= idx) then
            local sim = cos({ data[i], data[idx] })[1]
            if (sim > max_sim) then max_sim = sim; max_idx = i; end
        end
    end
    print(max_sim, max_idx)
end

local function print_top_k(sorted_scores, sorted_indices, idx_string_map, K, dictionary, en_only, es_only, tac_only)
    local k, i = 0, 1
    while (k < K  and i < sorted_indices:size(1)) do
        local ith_idx = sorted_indices[i]
        local token = idx_string_map[ith_idx] ~= nil and idx_string_map[ith_idx] or ith_idx
        token = dictionary[token] and token .. ' ' .. dictionary[token] .. '@es' or token
        if (not tac_only or string.sub(token, 0, 4) == "per:" or string.sub(token, 0, 4) == "org:") and
                (not es_only or string.sub(token, -3) == '@es') and (not en_only or string.sub(token, -3) ~= '@es') then
            print(token, sorted_scores[i])
            k = k + 1
        end
        i = i + 1
    end
end

local function k_nearest_neighbors_embeddings(input_token, embeddings, K, string_idx_map, idx_string_map, dictionary)
    embeddings = to_cuda(embeddings)
    local idx = string_idx_map[input_token]
    if idx ~= nil then
        local cos = to_cuda(nn.CosineDistance())
        local x = {embeddings[idx]:view(1, embeddings:size(2)):expandAs(embeddings), embeddings }
        local scores = cos(x)
        scores[idx] = 0

        local sorted_scores, sorted_indices = torch.sort(scores, true)
        if params.bothLang then
            print('En only')
            print_top_k(sorted_scores, sorted_indices, idx_string_map, K, dictionary, true, false, false)
            print('Es only')
            print_top_k(sorted_scores, sorted_indices, idx_string_map, K, dictionary, false, true, false)
        else
            print_top_k(sorted_scores, sorted_indices, idx_string_map, K, dictionary, params.enOnly, params.esOnly, params.tacOnly)
        end
    end
end

local function k_nearest_neighbors_encoder(input, encoder, K, string_idx_map, idx_string_map, token_count, dictionary)
    local input_table = {}

    -- dont split the input into tokens, take whole relation
    if params.relations then
        table.insert(input_table, string_idx_map[input])
    else
        -- split input phrase into tokens
        for token in string.gmatch(input, "[^" .. params.delim .. "]+") do
            table.insert(input_table, string_idx_map[token])
        end
    end

    if #input_table > 0 then
        -- convert input tokens to index tensor
        local input_tensor = torch.Tensor(input_table):view(1, #input_table)
        -- encode input tokens
        local encoded_input = encoder(input_tensor):clone()
        local cos = to_cuda(nn.CosineDistance())

        -- find max index in vocab
        local max_idx = 0
        for _, idx in pairs(string_idx_map) do
            if idx > max_idx then max_idx = idx end
        end

        local compare_tensor = to_cuda(torch.range(1,max_idx):view(max_idx, 1))
        local compare_encoded = encoder(compare_tensor)
        local scores = cos({compare_encoded, encoded_input:expandAs(compare_encoded) })
        local sorted_scores, sorted_indices = torch.sort(scores, true)
        if params.bothLang then
            print('En only')
            print_top_k(sorted_scores, sorted_indices, idx_string_map, K, dictionary, true, false, false)
            print('Es only')
            print_top_k(sorted_scores, sorted_indices, idx_string_map, K, dictionary, false, true, false)
        else
            print_top_k(sorted_scores, sorted_indices, idx_string_map, K, dictionary, params.enOnly, params.esOnly, params.tacOnly)
        end
    end
end


local token_count, string_idx_map, idx_string_map = load_map(params.map)
local dictionary = load_dictionary_file(params.dictionary, ' ')
local inputs = params.inputFile ~= '' and load_input_file(params.inputFile, '\t') or {params.input }

if params.model ~= '' then
    local model = torch.load(params.model)
    local text_encoder = to_cuda(model.text_encoder ~= nil and model.text_encoder or model.encoder)
    text_encoder:evaluate()
    for _, input in pairs(inputs) do
        print(input)
        k_nearest_neighbors_encoder(input, text_encoder, params.topK, string_idx_map, idx_string_map, token_count, dictionary)
    end
else
    local embeddings = torch.load(params.embeddings)
    for _, input in pairs(inputs) do
        print(input)
        k_nearest_neighbors_embeddings(input, embeddings, params.topK, string_idx_map, idx_string_map, dictionary)
    end
end

