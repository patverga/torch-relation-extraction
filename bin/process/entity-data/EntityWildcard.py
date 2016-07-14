import sys
import getopt


####
#    given a candidate file, replaces the entity surface forms with wild card $ARG tokens
#    by default keep whole sentence, if -x given - extracts the text between entities
####

def process_line(line):
    sentence_id, mention_start_idx, mention_length, mention_str, fb_id, sentence = line.strip().split('\t')
    start_idx = int(mention_start_idx)
    end_idx = start_idx+int(mention_length)

    start_token = '$MENTION_START'
    end_token = '$MENTION_END'

    tokens = sentence.split(' ')
    left = tokens[:start_idx]
    middle = tokens[start_idx:end_idx]
    right = tokens[end_idx:]

    wild_card_sentence = ' '.join(left + [start_token] + middle + [end_token] + right)
    return '\t'.join([sentence_id, mention_start_idx, mention_length, mention_str, fb_id, wild_card_sentence])


def main(argv):
    in_file = ''
    out_file = ''

    help_msg = 'CandidateWildCardArgs.py -i <inFile> -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["inFile=", "outFile="])
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

    print 'Processing lines from ' + in_file
    data = [process_line(line) for line in open(in_file, 'r')]

    print 'Exporting lines to ' + out_file
    out = open(out_file, 'w')
    [out.write(line + '\n') for line in data]
    out.close()

    print 'Done'


if __name__ == "__main__":
    main(sys.argv[1:])
