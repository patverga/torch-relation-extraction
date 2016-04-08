package.path = package.path .. ";src/?.lua"

require 'rnn'
require 'nn-modules/ViewTable'

local cmd = torch.CmdLine()
cmd:option('-relationMap', '', 'file of relations')
cmd:option('-tokenMap', '', 'file of token id map')
cmd:option('-model', '', 'trained model with a text encoder [optional, uses embeddings directly otherwise]')
cmd:option('-topK', 5, 'number of top nearest nieghbors to find')
cmd:option('-input', '', 'input token to find nearest neighbors for')
cmd:option('-inputFile', '', 'file of inputs to query')
cmd:option('-gpuid', -1, 'Which gpu to use, -1 for cpu (default)')
cmd:option('-delim', ' ', 'split tokens on this')
cmd:option('-esOnly', false, 'only return spanish relations')
cmd:option('-enOnly', false, 'only return english relations')
cmd:option('-tacOnly', false, 'only return tac relations')
cmd:option('-dictionary', '', 'check if tokens are in translation dictionary')

local params = cmd:parse(arg)
local function to_cuda(x) return params.gpuid >= 0 and x:cuda() or x end
if params.gpuid >= 0 then require 'cunn'; cutorch.manualSeed(0); cutorch.setDevice(params.gpuid + 1) else require 'nn' end

local function load_map(map_file)
    local token_count = 0
    local string_idx_map = {}
    local idx_string_map = {}
    for line in io.lines(map_file) do
        local token, idx = string.match(line, "([^\t]+)\t([^\t]+)")
        if token and idx then
            token_count = token_count + 1
            string_idx_map[token] = tonumber(idx)
            idx_string_map[tonumber(idx)] = token
        end
    end
    return token_count,  string_idx_map, idx_string_map
end

local function load_dictionary(dictionary_file)
    local dictionary = {}
    if params.dictionary ~= '' then
        for line in io.lines(dictionary_file) do
            local en, es = string.match(line, "([^ ]+) ([^ ]+)")
            if en and es then
                dictionary[en] = es
            end
        end
    end
    return dictionary
end

local function load_input_file(file, delim)
    local inputs = {}
    if file ~= '' then
        for line in io.lines(file) do
            if line then table.insert(inputs, line) end
        end
    end
    return inputs
end

local function print_top_k(sorted_scores, sorted_indices, embedding_idx_map, idx_string_map, K, dictionary)
    local k, i = 0, 1
    while (k < K  and i < sorted_indices:size(1)) do
        local ith_idx = embedding_idx_map[sorted_indices[i]]
        local relation = idx_string_map[ith_idx]
        -- is this a spanish relation
        local es = false
        local rel_tokens = {}
        for t in string.gmatch(relation, '([^ ]+)') do
            if #rel_tokens == 0 and string.sub(t, -3) == '@es' then es = true end
            -- this is a spanish word
            if string.sub(t, -3) == '@es' then t = string.sub(t, 0, -4) end
            -- this word is in the dictionary
            if dictionary[t] then t = dictionary[t] end
            table.insert(rel_tokens, t)
        end
        if (not params.esOnly or es) and (not params.enOnly or not es) then
            print(relation, sorted_scores[i])
            if es then
                local tranlated_relation = '';
                for i = 1, #rel_tokens do tranlated_relation = tranlated_relation .. rel_tokens[i] .. ' ' end
                print(tranlated_relation)
            end
            k = k + 1
        end
        i = i + 1
    end
end

local function k_nearest_neighbors_embeddings(input_token, embeddings, embedding_idx_map, index_embedding_map, K, string_idx_map, idx_string_map, dictionary)
    embeddings = to_cuda(embeddings)
    local idx = index_embedding_map[string_idx_map[input_token]]
    if idx ~= nil then
        local cos = to_cuda(nn.CosineDistance())
        local x = {embeddings[idx]:view(1, embeddings:size(2)):expandAs(embeddings), embeddings }
        local scores = cos(x)
        scores[idx] = 0

        local sorted_scores, sorted_indices = torch.sort(scores, true)
        if params.bothLang then
            print('En only')
            print_top_k(sorted_scores, sorted_indices, embedding_idx_map, idx_string_map, K, dictionary)
            print('Es only')
            print_top_k(sorted_scores, sorted_indices, embedding_idx_map, idx_string_map, K, dictionary)
        else
            print_top_k(sorted_scores, sorted_indices, embedding_idx_map, idx_string_map, K, dictionary)
        end
    end
end

local function encode_all_relations(encoder, rel_string_idx_map, token_string_idx_map, rel_total)
    local relations = {}
    local count = 0
    local max_len = 0
    local join = nn.JoinTable(1)
    -- convert relations to tokens and split by seq length for batching
    print ("Converting relations to tensors and batching")
    for rel, idx in pairs(rel_string_idx_map) do
        local rel_tokens = {}
        for t in string.gmatch(rel, '([^ ]+)') do table.insert(rel_tokens, token_string_idx_map[t] or 1) end
        local rel_tensor = torch.Tensor(rel_tokens)
        local len = rel_tensor:size(1)
        max_len = math.max(len, max_len)
        rel_tensor = rel_tensor:view(1, len)
        if relations[len] == nil then relations[len] = {rel_indices = {}, tensors = {}} end
        table.insert(relations[len].rel_indices, idx)
        table.insert(relations[len].tensors, rel_tensor)
        if count % 100 == 0 then io.write(string.format('\r%.2f percent complete', 100*(count / rel_total))); io.flush() end
        count = count + 1
    end

    print ("\nEncoding the joined tensors")
    local embedding_idx_map = {}
    local embeddings = {}
    for len = 1, max_len do
        io.write(string.format('\r%.2f percent complete', 100*(len / max_len))); io.flush()
        if relations[len] then
            local joined_tensor = join(relations[len].tensors)
            table.insert(embeddings, encoder(joined_tensor):clone())
            table.insert(embedding_idx_map, torch.Tensor(relations[len].rel_indices))
        end
    end
    collectgarbage()
    print ("\nJoining final results")
    embeddings = join(embeddings):clone()
    embedding_idx_map = join(embedding_idx_map):clone()

    local index_embedding_map = torch.Tensor(embedding_idx_map:max())
    for i = 1, embedding_idx_map:size(1) do
        index_embedding_map[embedding_idx_map[i]] = i
    end
    return embeddings, embedding_idx_map, index_embedding_map
end

print('Loading maps and model')
local inputs = params.inputFile ~= '' and load_input_file(params.inputFile, '\t') or {params.input }
local _, token_string_idx_map, _ = load_map(params.tokenMap)
local rel_count, rel_string_idx_map, rel_idx_string_map = load_map(params.relationMap)
-- add all inputs to relmaps
for _, input in pairs(inputs) do
    rel_count = rel_count + 1
    rel_string_idx_map[input] = rel_count
    rel_idx_string_map[rel_count] = input
end

local dictionary = load_dictionary(params.dictionary)
local model = torch.load(params.model)
local text_encoder = to_cuda(model.text_encoder ~= nil and model.text_encoder or model.encoder)
text_encoder:evaluate()
print ('Encoding all relations')
local embeddings, embedding_idx_map, index_embedding_map = encode_all_relations(text_encoder, rel_string_idx_map, token_string_idx_map, rel_count)

for _, input in pairs(inputs) do
    print ('\n\n'..input)
    k_nearest_neighbors_embeddings(input, embeddings, embedding_idx_map, index_embedding_map, params.topK, rel_string_idx_map, rel_idx_string_map, dictionary)
end




