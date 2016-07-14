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
    keep_singles = False

    help_msg = 'co-occuring-mentions-from-mtx.py -i <inFile> -o <outputfile> [-m <maxSamples> -s <keepSingle>]'
    try:
        opts, args = getopt.getopt(argv, "hi:o:m:s", ["inFile=", "outFile="])
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
        elif opt in ("-s", "--keepSingles"):
            keep_singles = True

    print 'Processing lines from ' + in_file
    entity_pairs = file_to_ep_dict(in_file)

    print 'Exporting lines to ' + out_file
    out = open(out_file, 'w')
    for entity_pair, rel_list in entity_pairs.iteritems():
        if keep_singles and len(rel_list) == 1:
            out.write(entity_pair[0] + '\t' + entity_pair[1] + '\t' + rel_list[0] + '\t' + rel_list[0] + '\t1\n')
        else:
            [out.write(entity_pair[0] + '\t' + entity_pair[1] + '\t' + rel_pair[0] + '\t' + rel_pair[1] + '\t1\n')
             for rel_pair in iter_sample_fast(itertools.combinations(rel_list, 2), max_samples)]
    out.close()
    print 'Done'


if __name__ == "__main__":
    main(sys.argv[1:])
