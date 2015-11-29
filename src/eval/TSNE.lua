require 'nn'
m = require 'manifold'

function load_dictionary_file(file, delim)
    local dictionary = {}
    if file ~= '' then
        for line in io.lines(file) do
            local en, es = string.match(line, "([^" .. delim .. "]+)" .. delim .. "([^" .. delim .. "]+)")
            if en ~= nil and es ~= nil then dictionary[en] = es end
        end
    end
    return dictionary
end

dict = load_dictionary_file('vocabs/no-log_min5_noramlized.en-es_dictionary-tokens.txt', '\t')
w = load_dictionary_file('tsne-file', '\t')
d = torch.load("models/2015-11-8/UniversalSchemaLSTM_no-log_min5_noramlized.en-es_dictionary/15-rel-weights")

labels = {}
embedding_table = {};
for label, idx in pairs(w) do
    embedding = d[idx]
    table.insert(embedding_table, embedding:view(1, embedding:size(1)));
    table.insert(labels, label);
end

embedding_matrix = nn.JoinTable(1)(embedding_table)
p = m.embedding.tsne(embedding_matrix, { dim = 2, perplexity = 30 })