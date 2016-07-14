--
-- User: pv
-- Date: 2/28/16
--

require 'torch'
require 'nn'

--[[
    Generates data for FB15k evaluations
]]--

local cmd = torch.CmdLine()
cmd:option('-inDir', '', 'directory containing all fb15k data (train.txt, valid.txt, test.txt)')
cmd:option('-outDir', '', 'output the new training files')
cmd:option('-vocabPrefix', '', 'vocab map prefix')

local params = cmd:parse(arg)

local function load_map(file_name)
    local map = {}
    print('Loading kb map : ' .. file_name)
    for line in io.lines(file_name) do
        local token, id = string.match(line, "(.*)\t([^\t]+)")
        if token and id then map[token] = tonumber(id) end
    end
    return map
end

local entity_map = load_map(params.vocabPrefix .. 'entities.txt')
local entpair_map = load_map(params.vocabPrefix .. 'entpairs.txt')
local rel_map = load_map(params.vocabPrefix .. 'relations.txt')
local token_map = load_map(params.vocabPrefix .. 'tokens.txt')


-- load all facts to filter false negatives
local facts = {}
for _, file in pairs({params.inDir .. '/train.txt', params.inDir .. '/valid.txt', params.inDir .. '/test.txt'}) do
    print ('Loading facts from : ' ..file)
    for line in io.lines(file) do
        local e1, e2, rel, _ = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
        local fact = e1..'\t'.. rel ..'\t' .. e2
        if not facts[fact] then facts[fact]=true end
    end
end

-- limit evaluation to entity pairs with textual mentions
local mention_eps = {}
local entities = {}
local file = params.inDir .. '/text_emnlp.txt'
local line_num = 0
print ('Loading mentions from : ' ..file)
for line in io.lines(file) do
    line_num = line_num+1;
    local e1, e2, _, _ = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
    if not entities[e1] then entities[e1]=true end
    if not entities[e2] then entities[e2]=true end
    local ep = e1..'\t'..e2
    if not mention_eps[ep] then mention_eps[ep]=true end
    if line_num % 50000 == 0 then io.write('\rline number: '..line_num); io.flush(); collectgarbage() end
end

local function insert_to_out_table(out_table, e1_idx, e2_idx, ep_idx, rel_idx, token_idx, label)
    table.insert(out_table.e1, e1_idx);         table.insert(out_table.e2, e2_idx)
    table.insert(out_table.ep, ep_idx);         table.insert(out_table.rel, rel_idx)
    table.insert(out_table.seq, token_idx);     table.insert(out_table.label, label)
end

local example_table
local e1, e1_idx, e2, e2_idx, neg_idx, rel, rel_idx, token_idx
for _, file in pairs({'valid', 'test'}) do
    local file_data_table = {}
    os.execute("mkdir -p " .. params.outDir..'/'..file)
    local filtered_facts = 0
    local filtered_ep = 0
    local kept_facts = 0
    local kept_ep = 0
    local line_num = 1
    local file_suffix = 0
    for line in io.lines(params.inDir..'/'..file..'.txt') do
        example_table = {e1={}, e2={}, ep={}, rel={}, seq={}, label={}}
        io.write(string.format('\rProcessing %s - line : %d \t kept ep : %d, filtered ep : %d \t kept facts : %2.1fK, filtered facts : %2.1fK',
            file, line_num, kept_ep, filtered_ep, kept_facts/1000, filtered_facts/1000))
        io.flush(); line_num = line_num+1
        e1, e2, rel, _ = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
        local ep = e1..'\t'..e2
        -- only keep ep if it had a textual mention
        if mention_eps[ep] then
            local ep_idx = entpair_map[ep]
            if ep_idx then
                kept_ep = kept_ep + 1
                -- positive fact
                e1_idx = entity_map[e1]
                e2_idx = entity_map[e2]
                rel_idx = rel_map[rel]
                token_idx = token_map[rel]
                insert_to_out_table(example_table, e1_idx, e2_idx, ep_idx, rel_idx, token_idx, 1)

                -- replace tail
                local head_fact = e1..'\t'.. rel
                for neg_ent, _ in pairs(entities) do
                    local neg_ep = e1..'\t'..neg_ent
                    -- only keep ep if it had a textual mention
                    if e1 ~= neg_ent and mention_eps[neg_ep] then
                        local neg_ep_idx = entpair_map[neg_ep]
                        -- make sure ep occured in a our training set
                        if neg_ep_idx then
                            local neg_fact = head_fact..'\t'..neg_ent
                            -- filter false negatives
                            if facts[neg_fact] then filtered_facts = filtered_facts +1
                            else
                                neg_idx = entity_map[neg_ent]
                                insert_to_out_table(example_table, e1_idx, neg_idx, neg_ep_idx, rel_idx, token_idx, 0)
                            end
                        end
                    end
                end
                example_table.e1 = torch.Tensor(example_table.e1):long():view(-1,1)
                example_table.e2 = torch.Tensor(example_table.e2):long():view(-1,1)
                example_table.ep = torch.Tensor(example_table.ep):long():view(-1,1)
                example_table.rel = torch.Tensor(example_table.rel):long():view(-1,1)
                example_table.seq = torch.Tensor(example_table.seq):long():view(-1,1)
                example_table.label = torch.Tensor(example_table.label):long():view(-1,1)
                kept_facts = kept_facts + example_table.label:size(1)

                table.insert(file_data_table, example_table)
            else
                print ('entity pair missing from vocab : ' .. ep)
            end
            else filtered_ep = filtered_ep + 1
        end

        if line_num % 100 == 0 then collectgarbage() end
    end
    print('\nSaving file ' .. file_suffix)
    torch.save(params.outDir .. '/' .. file .. '/' .. file_suffix .. '.torch', file_data_table)
    file_data_table = {}; file_suffix = file_suffix +1
end