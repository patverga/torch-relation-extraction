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


local entities = {}
local facts = {}
for _, file in pairs({params.inDir .. '/train.txt', params.inDir .. '/valid.txt', params.inDir .. '/test.txt'}) do
    print ('Loading facts from : ' ..file)
    for line in io.lines(file) do
        local e1, e2, rel, _ = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
        if not entities[e1] then table.insert(entities, e1) end
        if not entities[e2] then table.insert(entities, e2) end
        local fact = e1..'\t'.. rel ..'\t' .. e2
        if not facts[fact] then facts[fact] = true end
    end
end

local function insert_to_out_table(out_table, e1_idx, e2_idx, ep_idx, rel_idx, token_idx, label)
    table.insert(out_table.e1, e1_idx)
    table.insert(out_table.e2, e2_idx)
    table.insert(out_table.ep, ep_idx)
    table.insert(out_table.rel, rel_idx)
    table.insert(out_table.seq, token_idx)
    table.insert(out_table.label, label)
end

local example_table
local e1, e1_idx, e2, e2_idx, neg_idx, rel, rel_idx, token_idx
--for _, file in pairs({ 'valid', 'test'}) do
for _, file in pairs({ 'valid', 'test'}) do
    local file_data_table = {}
    os.execute("mkdir -p " .. params.outDir..'/'..file)
    local filtered_facts = 0
    local kept_facts = 0
    local line_num = 1
    local file_suffix = 0
    for line in io.lines(params.inDir..'/'..file..'.txt') do
        example_table = {e1={}, e2={}, ep={}, rel={}, seq={}, label={}}
        io.write('\rProcessing '..file ..'.txt line : ' .. line_num ..'\t filtered facts : ' .. filtered_facts/1000  .. ' k \t new facts : '..kept_facts/1000000 ..'m' )
        io.flush(); line_num = line_num+1
        e1, e2, rel, _ = string.match(line, "([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)")
        local ep = e1..'\t'..e2
        local ep_idx = entpair_map[ep]
        if ep_idx then
            e1_idx = entity_map[e1]
            e2_idx = entity_map[e2]
            rel_idx = rel_map[rel]
            token_idx = token_map[rel]
            insert_to_out_table(example_table, e1_idx, e2_idx, ep_idx, rel_idx, token_idx, 1)

--            -- replace head
--            local tail_fact = rel..'\t'.. e2
--            for _, neg_ent in pairs(entities) do
--                local neg_ep_idx = entpair_map[neg_ent..'\t'..e2]
--                if neg_ep_idx then
--                    local neg_fact = neg_ent..'\t'..tail_fact
--                    if facts[neg_fact] then filtered_facts = filtered_facts +1
--                    else
--                        neg_idx = entity_map[neg_ent]
--                        insert_to_out_table(out_table, neg_idx, e2_idx, neg_ep_idx, rel_idx, token_idx, 0)
--                    end
--                end
--            end
            -- replace tail
            local head_fact = e1..'\t'.. rel
            for _, neg_ent in pairs(entities) do
                local neg_fact = head_fact ..'\t'..neg_ent
                local neg_ep_idx = entpair_map[e1..'\t'..neg_ent]
                if neg_ep_idx then
                    if facts[neg_fact] then filtered_facts = filtered_facts +1
                    else
                        neg_idx = entity_map[neg_ent]
                        insert_to_out_table(example_table, e1_idx, neg_idx, neg_ep_idx, rel_idx, token_idx, 0)
                    end
                end
            end
            example_table.e1 = torch.Tensor(example_table.e1):long():view(-1,1)
            example_table.e2 = torch.Tensor(example_table.e2):long():view(-1,1)
            example_table.ep = torch.Tensor(example_table.ep):long():view(-1,1)
            example_table.rel = torch.Tensor(example_table.rel):long():view(-1,1)
            example_table.seq = torch.Tensor(example_table.seq):long():view(-1,1)
            example_table.label = torch.Tensor(example_table.label):long():view(-1,1)
--            print (out_table)
            kept_facts = kept_facts + example_table.label:size(1)

--            local fact_str = e1_idx..'_'..e2_idx..'_'..rel_idx
--            local out_file =(fact_str:gsub("/", "-"))
--            out_file = params.outDir .. '/' .. file .. '/' .. out_file
            table.insert(file_data_table, example_table)
--            print(out_table); out_table = nil;
        end
        if line_num % 25 == 0 then collectgarbage() end
        if line_num % 400 == 0 then
            print('\nSaving file ' .. file_suffix)
            torch.save(params.outDir .. '/' .. file .. '/' .. file_suffix .. '.torch', file_data_table)
            file_data_table = {}; file_suffix = file_suffix +1
        end
    end
end