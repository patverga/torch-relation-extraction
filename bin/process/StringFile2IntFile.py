__author__ = 'pat'

import re
import sys
import getopt
import pickle
from collections import defaultdict


###
# Takes a 4 col mtx file and maps string entries to int id's
# use -v to export string-int maps
###

def process_line(chars, ent_map, ep_map, line, rel_map, token_counter, double_vocab, replace_digits):
    e1_str, e2_str, rel_str, label = line.strip().split('\t')
    ep_str = e1_str + '\t' + e2_str

    # don't separate tac relations into characters, keep as single token
    if str.startswith(rel_str, 'per:') or str.startswith(rel_str, 'org:'):
        tokens = [rel_str]
    else:
        # normalize digits except in $ARG wildcard tokens
        if replace_digits:
            rel_str = re.sub(r'(?<!\$ARG)[0-9]', '#', rel_str)
        tokens = rel_str.split(' ')

        # split words into char tokens except leave $ARG as single tokens, flatten to list
        if chars:
            tokens = [ch for tok in
                      [[t, ' '] if str.startswith(t, '$ARG') else list(t + ' ') for t in tokens]
                      for ch in tok]
            # we added an extra ' ' to the end TODO handle this better
            if len(tokens) > 0:
                del(tokens[-1])

        # have seperate vocabularies for when arg1 proceeds arg2 and vice-versa
        if double_vocab and len(tokens) > 1 \
                and "$ARG1" in tokens and "$ARG2" in tokens \
                and tokens.index("$ARG1") > tokens.index("$ARG2"):
            tokens = [token + '_ARG2' for token in tokens]

    for token in tokens:
        token_counter[token] += 1

    # if not chars:
    rel_str = ' '.join(tokens)

    # add 1 for 1 indexing
    ent_map.setdefault(e1_str, str(len(ent_map) + 1))
    ent_map.setdefault(e2_str, str(len(ent_map) + 1))
    ep_map.setdefault(ep_str, str(len(ep_map) + 1))
    rel_map.setdefault(rel_str, str(len(rel_map) + 1))
    return e1_str, e2_str, ep_str, rel_str, tokens, label


def export_line(e1_str, e2_str, ep_str, rel_str, tokens, ent_map, ep_map, rel_map, token_map, label, out):
    # map tokens
    token_ids = [str(token_map[token]) if token in token_map else '1' for token in tokens]
    e1 = ent_map[e1_str]
    e2 = ent_map[e2_str]
    ep = ep_map[ep_str]
    rel = rel_map[rel_str]
    out.write('\t'.join([e1, e2, ep, rel, ' '.join(token_ids), label]) + '\n')


def export_map(file_name, vocab_map):
    with open(file_name, 'w') as fp:
        vocab_map = {token: int(int_id) for token, int_id in vocab_map.iteritems()}
        for token in sorted(vocab_map, key=vocab_map.get, reverse=False):
            fp.write(token + '\t' + str(vocab_map[token]) + '\n')


def main(argv):
    in_file = ''
    out_file = ''
    save_vocab_file = ''
    load_vocab_file = ''
    chars = False
    min_count = 0
    max_seq = sys.maxint
    double_vocab = False
    reset_tokens = False
    replace_digits = False

    help_msg = 'test.py -i <inFile> -o <outputfile> -m <throw away tokens seen less than this many times> \
-s <throw away relations longer than this> -c <use char tokens (default is use words)> -d <double vocab depending on if [A1 rel A2] or [A2 rel A1]>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:dcm:s:l:v:rn", ["inFile=", "outFile=", "saveVocab=", "loadVocab=",
                                                               "chars", "doubleVocab", "minCount=", "maxSeq=",
                                                               "resetVocab", "noNumbers"])
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt in ("-i", "--inFile"):
            in_file = arg
        elif opt in ("-o", "--outFile"):
            out_file = arg
        elif opt in ("-m", "--minCount"):
            min_count = int(arg)
        elif opt in ("-s", "--maxSeq"):
            max_seq = int(arg)
        elif opt in ("-c", "--chars"):
            chars = True
        elif opt in ("-v", "--saveVocab"):
            save_vocab_file = arg
        elif opt in ("-l", "--loadVocab"):
            load_vocab_file = arg
        elif opt in ("-d", "--doubleVocab"):
            double_vocab = True
        elif opt in ("-r", "--resetVocab"):
            reset_tokens = True
        elif opt in ("-n", "--noNumbers"):
            replace_digits = True
    print 'Input file is :', in_file
    print 'Output file is :', out_file
    print 'Exporting char tokens' if chars else 'Exporting word tokens'
    if chars and double_vocab:
        print 'Double vocab and chars not compatible, setting double vocab to false'
        double_vocab = False

    # load memory maps from file or initialize new ones
    if load_vocab_file:
        with open(load_vocab_file, 'rb') as fp:
            [ent_map, ep_map, rel_map, token_map, token_counter] = pickle.load(fp)
        if reset_tokens:
            # this should probably be a different flag
            rel_map = {}
            token_map = {}
            token_counter = defaultdict(int)
    else:
        ent_map = {}
        ep_map = {}
        rel_map = {}
        token_map = {}
        token_counter = defaultdict(int)

    # memory map all the data and return processed lines
    print 'Processing lines and getting token counts'
    data = [process_line(chars, ent_map, ep_map, line, rel_map, token_counter, double_vocab, replace_digits)
            for line in open(in_file, 'r')]

    # prune infrequent tokens
    if reset_tokens or not load_vocab_file:
        filtered_tokens = {token: count for token, count in token_counter.iteritems() if count > min_count}
        sorted_tokens = [token for token in sorted(filtered_tokens, key=filtered_tokens.get, reverse=True)]
        token_map = {token: i + 1 for i, token in enumerate(sorted_tokens)}

    print 'Exporting processed lines to file'
    # export processed data
    out = open(out_file, 'w')
    [export_line(e1_str, e2_str, ep_str, rel_str, tokens, ent_map, ep_map, rel_map, token_map, label, out)
     for e1_str, e2_str, ep_str, rel_str, tokens, label in data if len(tokens) <= max_seq]
    out.close()
    print 'Num ents: ', len(ent_map), 'Num eps: ', len(ep_map), \
        'Num rels: ', len(rel_map), 'Num tokens: ', len(token_map)

    if save_vocab_file:
        print 'Exporting vocab maps to file'
        with open(save_vocab_file, 'wb') as fp:
            pickle.dump([ent_map, ep_map, rel_map, token_map, token_counter], fp)
        export_map(save_vocab_file + '-tokens.txt', token_map)
        export_map(save_vocab_file + '-relations.txt', rel_map)
        export_map(save_vocab_file + '-entities.txt', ent_map)
        export_map(save_vocab_file + '-entpairs.txt', ep_map)


if __name__ == "__main__":
    main(sys.argv[1:])
