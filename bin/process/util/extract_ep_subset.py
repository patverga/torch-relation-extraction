from __future__ import print_function
import sys

###
#  given two 4 col mtx files, takes a subset of file 2 where each
#  entity pair is contained in file 1
###

fname1 = sys.argv[1]
fname2 = sys.argv[2]
output = sys.argv[3]

# get number of fields in first line of first file, then
# expect every line in both files to contain this many fields
with open(fname1) as f:
    num_fields = len(f.readline().strip().split("\t"))

# read first file into map
print("reading entity pairs from first file")
entity_pairs = set()
with open(fname1) as f:
    for line in f:
        fields = line.strip().split("\t")
        assert len(fields) == num_fields, "Wrong number of fields in line"
        entity_pairs.add((fields[0], fields[1]))

print("read %d unique entity pairs" % len(entity_pairs))
print("matching entity pairs from second file and writing")
out_file = open(output, 'w')
matched = set()
with open(fname2) as f:
    for line in f:
        fields = line.strip().split("\t")
        assert len(fields) == num_fields, "Wrong number of fields in line"
        if (fields[0], fields[1]) in entity_pairs:
            matched.add((fields[0], fields[1]))
            print(fields[0] + "\t" + fields[1] + "\t" + fields[2] + "\t1", file=out_file)
out_file.close()
print("matched %d unique entity pairs" % len(matched))
