__author__ = 'pat'

import re
import sys
import getopt
import pickle
from collections import defaultdict


###
# Takes a 3 col mtx file and maps string entries to int id's
# use -v to export string-int maps
###

def process_seq(seq, token_counter, replace_digits, chars, double_vocab):
    # normalize digits except in $ARG wildcard tokens
    if replace_digits:
        seq = re.sub(r'(?<!\$ARG)[0-9]', '#', seq)
    tokens = seq.split(' ')

    # split words into char tokens except leave $ARG as single tokens, flatten to list
    if chars:
        tokens = [ch for tok in
                  [[t, ' '] if str.startswith(t, '$ARG') else list(t + ' ') for t in tokens]
                  for ch in tok]
        # we added an extra ' ' to the end TODO handle this better
        if len(tokens) > 0:
            del (tokens[-1])

    # have seperate vocabularies for when arg1 proceeds arg2 and vice-versa
    if double_vocab and len(tokens) > 1 \
            and "$ARG1" in tokens and "$ARG2" in tokens \
            and tokens.index("$ARG1") > tokens.index("$ARG2"):
        tokens = [token + '_ARG2' for token in tokens]

    for token in tokens:
        token_counter[token] += 1

    return tokens


def process_line(line, col_str_map, row_str_map, col_token_counter, row_token_counter, double_vocab, replace_digits,
                 chars):
    row_seq, col_seq, label = line.strip().split('\t')

    row_tokens = process_seq(row_seq, row_token_counter, replace_digits, chars, double_vocab)
    col_tokens = process_seq(col_seq, col_token_counter, replace_digits, chars, double_vocab)
    col_str = ' '.join(col_tokens)
    row_str = ' '.join(row_tokens)

    # add 1 for 1 indexing
    col_str_map.setdefault(col_str, str(len(col_str_map) + 1))
    row_str_map.setdefault(row_str, str(len(row_str_map) + 1))

    return row_tokens, col_tokens, row_str, col_str, label


def export_line(row_tokens, col_tokens, row_str, col_str, row_token_map, col_token_map, row_str_map, col_str_map, label, out):
    # map tokens -- sets unk idx to 1
    col_token_ids = [str(col_token_map[token]) if token in col_token_map else '1' for token in col_tokens]
    row_token_ids = [str(row_token_map[token]) if token in row_token_map else '1' for token in row_tokens]
    cs = col_str_map[col_str]
    rs = row_str_map[row_str]
    out.write('\t'.join([rs, ' '.join(row_token_ids), cs, ' '.join(col_token_ids), label]) + '\n')


def export_map(file_name, vocab_map):
    with open(file_name, 'w') as fp:
        vocab_map = {token: int(int_id) for token, int_id in vocab_map.iteritems()}
        for token in sorted(vocab_map, key=vocab_map.get, reverse=False):
            fp.write(token + '\t' + str(vocab_map[token]) + '\n')


def filter_tokens(token_counter, min_count):
    # prune infrequent tokens - sets unkidx to 1
    filtered_tokens = {token: count for token, count in token_counter.iteritems() if count > min_count}
    sorted_tokens = [token for token in sorted(filtered_tokens, key=filtered_tokens.get, reverse=True)]
    token_map = {token: i + 3 for i, token in enumerate(sorted_tokens)}
    return token_map


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
    merge_maps = False

    help_msg = 'test.py -i <inFile> -o <outputfile> -m <throw away tokens seen less than this many times> \
-s <throw away relations longer than this> -c <use char tokens (default is use words)> -d <double vocab depending on if [A1 rel A2] or [A2 rel A1]>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:dcm:s:l:v:rng", ["inFile=", "outFile=", "saveVocab=", "loadVocab=",
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
        elif opt in ("-g", "--mergeMaps"):
            merge_maps = True
    print 'Input file is :', in_file
    print 'Output file is :', out_file
    print 'Exporting char tokens' if chars else 'Exporting word tokens'
    if chars and double_vocab:
        print 'Double vocab and chars not compatible, setting double vocab to false'
        double_vocab = False

    # load memory maps from file or initialize new ones
    if load_vocab_file:
        with open(load_vocab_file, 'rb') as fp:
            [col_str_map, row_str_map, col_token_map, row_token_map, col_token_counter, row_token_counter] = pickle.load(fp)
        if reset_tokens:
            # this should probably be a different flag
            col_str_map = {}
            col_token_map = {}
            col_token_counter = defaultdict(int)
    else:
        col_str_map = {}
        row_str_map = {}
        col_token_map = {}
        row_token_map = {}
        col_token_counter = defaultdict(int)
        row_token_counter = defaultdict(int)

    if merge_maps:
        col_token_counter = row_token_counter
        col_token_map = row_token_map
        col_str_map = row_str_map

    # memory map all the data and return processed lines
    print 'Processing lines and getting token counts'
    data = [
        process_line(line, col_str_map, row_str_map, col_token_counter, row_token_counter, double_vocab, replace_digits,
                     chars)
        for line in open(in_file, 'r')]

    if reset_tokens or not load_vocab_file:
        col_token_map = filter_tokens(col_token_counter, min_count)
        row_token_map = filter_tokens(row_token_counter, min_count)

    col_token_map['<<UNK>>'] = 1; col_token_map['<<PAD>>'] = 2
    row_token_map['<<UNK>>'] = 1; row_token_map['<<PAD>>'] = 2

    print 'Exporting processed lines to file'
    # export processed data
    out = open(out_file, 'w')
    [export_line(row_tokens, col_tokens, row_str, col_str, row_token_map, col_token_map, row_str_map, col_str_map, label, out)
      for row_tokens, col_tokens, row_str, col_str, label in data if len(row_tokens) <= max_seq and len(row_tokens) <= max_seq]
    out.close()

    print 'Num rows: ', len(row_str_map), 'Num row tokens: ', len(row_token_map), \
        'Num cols: ', len(col_str_map), 'Num col tokens: ', len(col_token_map)

    if save_vocab_file:
        print 'Exporting vocab maps to file'
        with open(save_vocab_file, 'wb') as fp:
            pickle.dump([col_str_map, row_str_map, col_token_map, row_token_map, col_token_counter, row_token_counter], fp)
        export_map(save_vocab_file + '-col-tokens.txt', col_token_map)
        export_map(save_vocab_file + '-row-tokens.txt', row_token_map)
        export_map(save_vocab_file + '-cols.txt', col_str_map)
        export_map(save_vocab_file + '-rows.txt', row_str_map)


if __name__ == "__main__":
    main(sys.argv[1:])
