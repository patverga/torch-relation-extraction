import sys
import getopt
import itertools
import random


def file_to_ep_dict(fname):
    entity_pairs = {}
    with open(fname) as f:
        for line in f:
            fields = line.strip().split("\t")
            ep = (fields[0], fields[1])
            if ep not in entity_pairs:
                entity_pairs[ep] = []
            entity_pairs[ep].append(fields[2])
    return entity_pairs


# for entity pairs with many mentions, randomly subsample the combinations
def iter_sample_fast(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    try:
        for _ in xrange(samplesize):
            results.append(iterator.next())
    except StopIteration:
        random.shuffle(results)
        return results
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


def main(argv):
    in_file = ''
    out_file = ''
    max_samples = 1000

    help_msg = 'CandidateWildCardArgs.py -i <inFile> -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:m:", ["inFile=", "outFile="])
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
        elif opt in ("-m", "--maxSamples"):
            max_samples = int(arg)

    print 'Processing lines from ' + in_file
    entity_pairs = file_to_ep_dict(in_file)

    print 'Exporting lines to ' + out_file
    out = open(out_file, 'w')
    for key, value in entity_pairs.iteritems():
        [out.write('____\t' + pair[0] + '\t' + pair[1] + '\t1\n') for pair in
         iter_sample_fast(itertools.combinations(value, 2), max_samples)]
    out.close()
    print 'Done'


if __name__ == "__main__":
    main(sys.argv[1:])
