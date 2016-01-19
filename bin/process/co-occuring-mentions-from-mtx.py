import sys
import getopt
import itertools


def file_to_ep_dict(fname):
    entity_pairs = {}
    with open(fname) as f:
        for line in f:
            fields = line.strip().split("\t")
            ep = (fields[0], fields[1])
            if ep not in entity_pairs:
                entity_pairs[ep] = []
            entity_pairs[ep] += fields[3]
    return entity_pairs


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
    entity_pairs = file_to_ep_dict(in_file)

    print 'Exporting lines to ' + out_file
    out = open(out_file, 'w')
    for key, value in entity_pairs.iteritems():
        [out.write('____\t' + pair[0] + '\t' + pair[1] + '\t1\n') for pair in itertools.product(value, repeat=2)]
    out.close()
    print 'Done'


if __name__ == "__main__":
    main(sys.argv[1:])
