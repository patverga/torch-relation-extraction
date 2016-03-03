import sys
import getopt
from collections import defaultdict


def main(argv):
    in_file = ''
    out_dir = ''
    map_file = ''
    help_msg = 'match-entity-pair-types.py -i <inFile> -o <out> -m <enttiy pair map>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:m:", ["inFile=", "out="])
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt in ("-i", "--inFile"):
            in_file = arg
        elif opt in ("-o", "--out"):
            out_dir = arg
        elif opt in ("-m", "--entitypair-map"):
            map_file = arg

    entity_type_map = {}
    type_int_map = {}

    def process_in_line():
        entity, entity_types = line.strip().split('\t')
        type_strings = entity_types.split(',')
        [type_int_map.setdefault(t, str(len(type_int_map) + 1)) for t in type_strings]
        entity_type_map[entity] = [type_int_map[t_str] for t_str in type_strings]


    type_ep_map = defaultdict(set)
    ep_type_int_map = {}
    ep_type_map = defaultdict(set)

    def process_map_line(i):
        e1, e2, idx = line.strip().split('\t')
        if e1 in entity_type_map and e2 in entity_type_map:
            e1_types = entity_type_map[e1]
            e2_types = entity_type_map[e2]
            # ep = e1 + '\t' + e2
            ep = idx
            ep_types = [t1 + ":" + t2 for t1 in e1_types for t2 in e2_types]
            [ep_type_int_map.setdefault(t, str(len(ep_type_int_map) + 1)) for t in ep_types]
            ep_types = [ep_type_int_map[t] for t in ep_types]
            ep_type_map[ep] = ep_types
            [type_ep_map[t].add(ep) for t in ep_types]
            if i % 100 == 0:
                sys.stdout.write("\r" + str(i) + "\t" + str(len(ep_types)) + "\t" + str(len(type_ep_map)))
                sys.stdout.flush()

    def export_map(k, values, o, i):
        sys.stdout.write("\r" + str(i))
        sys.stdout.flush()
        out.write(k + '\t')
        out.write(','.join(values) + '\n')

    print 'Reading in data from ' + map_file
    [process_in_line() for line in open(in_file, 'r')]
    [process_map_line(i) for i, line in enumerate(open(map_file, 'r'))]
    print '\nexporting ep-type map'
    out = open(out_dir+'/ep_type.map', 'w')
    [export_map(ep, types, out, i) for i, (ep, types) in enumerate(ep_type_map.iteritems())]
    out.close()
    print '\nexporting type-ep map'
    out = open(out_dir+'/type_ep.map', 'w')
    [export_map(t, eps, out, i) for i, (t, eps) in enumerate(type_ep_map.iteritems())]
    out.close()
    print '\nexporting ep_type-int map'
    out = open(out_dir+'/ep_type_int.map', 'w')
    [export_map(t, [idx], out, i) for i, (t, idx) in enumerate(ep_type_int_map.iteritems())]
    out.close()
    # print '\nexporting ep-ep map'
    # out = open(out_dir+'/ep_ep.map', 'w')
    # [export_map(ep, set([ep2 for t in types for ep2 in type_ep_map[t]]), out, i) for i, (ep, types) in enumerate(ep_type_map.iteritems())]
    # out.close()
    print '\nDone'


if __name__ == "__main__":
    main(sys.argv[1:])
