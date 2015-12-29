import sys
import getopt

####
#    given a candidate file, replaces the entity surface forms with wild card $ARG tokens
####

def process_line(line):
    arg1, tac_rel, arg2, doc_info, s1_str, e1_str, s2_str, e2_str, sentence = line.strip().split('\t')
    s1 = int(s1_str)
    s2 = int(s2_str)
    e1 = int(e1_str)
    e2 = int(e2_str)

    if s1 < s2:
        arg1_str = '$ARG1'
        arg2_str = '$ARG2'
    else:
        arg1_str = '$ARG2'
        arg2_str = '$ARG1'
        s1, s2, e1, e2 = s2, s1, e2, e1

    tokens = sentence.split(' ')
    left = tokens[:s1]
    middle = tokens[e1:s2]
    right = tokens[e2:]

    wild_card_sentence = ' '.join(left + [arg1_str] + middle + [arg2_str] + right)
    return '\t'.join([arg1, tac_rel, arg2, doc_info, s1_str, e1_str, s2_str, e2_str, wild_card_sentence])


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
