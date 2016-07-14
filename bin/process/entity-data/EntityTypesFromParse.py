import sys
import getopt
import json
import gzip
from collections import defaultdict
import codecs


def main(argv):
    in_file = ''
    out_file = ''
    entity_file = ''

    help_msg = 'EntityTypesFromParse.py -i <inFile> -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:e:", ["inFile=", "outFile=", "entityFile="])
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
        elif opt in ("-e", "--entityFile"):
            entity_file = arg

    print 'Reading entity spans'
    sentence_mentions = defaultdict(list)
    for i, line in enumerate(open(entity_file, 'r')):
        sent_id, mention_start_idx, mention_length, mention_string, fb_id = line.strip().split('\t')
        sentence_mentions[sent_id].append((mention_start_idx, mention_length, mention_string, fb_id))
        if i % 1000 == 0:
            sys.stdout.write('{0} lines read\r'.format(i))
            sys.stdout.flush()

    print 'Processing lines from ' + in_file
    out = codecs.open(out_file, 'w', 'utf-8')
    for line in iter(gzip.open(in_file, 'rb')):
        parsed_data = json.loads(line)
        sent_id = parsed_data['docName']
        data = parsed_data['sentences'][0]
        tokens = [token.encode('utf-8') for token in data['tokenStrings']]
        parse_parents = [int(i) for i in data['parseParents']]
        parse_labels = data['parseLabels']
        pos = data['posLabels']
        for mention_start_idx_str, mention_length_str, mention_string, fb_id in sentence_mentions[sent_id]:
            mention_start_idx, mention_length = int(mention_start_idx_str), int(mention_length_str)
            mention_end_idx = mention_start_idx + mention_length
            # print '\n' + str(list(enumerate(tokens)))
            # print parse_parents
            parent_idx = parse_parents[mention_start_idx]
            # print parent_idx, mention_start_idx, mention_length, mention_string, fb_id

            current_path = ''
            while parent_idx > -1:
                if parent_idx < mention_start_idx or parent_idx > mention_end_idx:
                    current_path = tokens[parent_idx] + ' ' + current_path
                    if pos[parent_idx].startswith('N') or pos[parent_idx].startswith('A') \
                            or pos[parent_idx].startswith('J') or pos[parent_idx].startswith('V'):
                        # print(parent_idx, parse_parents[parent_idx], pos[parent_idx], parse_labels[parent_idx], tokens[parent_idx])
                        # print current_path
                        out_str = sent_id + '\t' + mention_start_idx_str + '\t' + mention_length_str + '\t' + mention_string + '\t' + fb_id + '\t' + current_path.decode('utf-8') + '\n'
                        # print(out_str)
                        out.write(out_str)
                parent_idx = -1 if pos[parent_idx].startswith('V') else parse_parents[parent_idx]





                # print 'Exporting lines to ' + out_file
    # [out.write(line + '\n') for line in data]
    out.close()

    print 'Done'


if __name__ == "__main__":
    main(sys.argv[1:])
