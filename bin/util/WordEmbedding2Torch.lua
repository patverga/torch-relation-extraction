--
-- User: pat
-- Date: 9/17/15
--


require 'torch'

--[[
    Takes a word embedding file and a vocab2index map file and exports a tac
    tensor containing all words from the vocab map initialized from the word embedding
    file when it exists, or randomly in [-.1,.1] otherwise
]]--
local cmd = torch.CmdLine()
cmd:option('-embeddingFile', '', 'input word embedding file')
cmd:option('-outFile', '', 'output for torch word embedding')
cmd:option('-delim', ' ', 'delimiter to break string on')
cmd:option('-vocabFile', ' ', 'txt file containing vocab-index map')
cmd:option('-dim', 0, 'dimension of embeddings - maybe dont need')
cmd:option('-filterId', 1, 'filter tokens mapping to unk id')
local params = cmd:parse(arg)


local function load_vocab(vocab_file)
    local vocab_map = {}
    local token_count = 0
    for line in io.lines(vocab_file) do
        local token, id = string.match(line, "([^\t]+)\t([^\t]+)")
        if token then
            id = tonumber(id)
            if id ~= params.filterId then
                vocab_map[token] = id
                if id > token_count then token_count = id end
            end
        end
    end
    return vocab_map, token_count
end

--- process a single line from an embedding file ---
local function process_line(line, vocab_map)
    -- assuming w2v format for now
    local idx = 0
    local token
    local values = {}
    for x in string.gmatch(line, "[^" .. params.delim .. "]+") do
        if idx == 0 then token = x
        elseif vocab_map[token] == nil then return nil, nil
        else table.insert(values, x)
        end
        idx = idx + 1
    end
    return token, torch.Tensor(values)
end

--- process the embedding file and convert to torch ---
local function process_file(embedding_file, vocab_map, token_count)
    local line_num = 0
    local matches = 0
    local embedding_tensor = torch.rand(token_count, params.dim):add(-.5):mul(0.1)
    print('Processing data')
    for line in io.lines(embedding_file) do
        local token, word_embedding = process_line(line, vocab_map)
        if token then
            if vocab_map[token] then
                matches = matches + 1
                local id = vocab_map[token]
                embedding_tensor[id] = word_embedding
            end
        end
        line_num = line_num + 1
        if line_num % 1000 == 0 then io.write('\rline : ' .. line_num); io.flush() end
    end
    print ('\rProcessed ' .. line_num .. ' lines and found ' .. matches .. ' matches.')
    return embedding_tensor
end

---- main
local vocab_map, token_count = load_vocab(params.vocabFile)
local embedding_tensor = process_file(params.embeddingFile, vocab_map, token_count)
torch.save(params.outFile, embedding_tensor)