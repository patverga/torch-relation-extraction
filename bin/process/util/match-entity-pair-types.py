import sys
import getopt
from collections import defaultdict
import gzip


def main(argv):
    type_map_file = ''
    entity_map_file = ''
    entity_pair_map_file = ''
    out_file = ''
    help_msg = 'match-entity-pair-types.py -i <inFile> -o <out> -m <enttiy pair map>'
    try:
        opts, args = getopt.getopt(argv, "ht:o:e:p:", ["inFile=", "out="])
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt in ("-t", "--type-map"):
            type_map_file = arg
        elif opt in ("-e", "--entity-map"):
            entity_map_file = arg
        elif opt in ("-p", "--entitypair-map"):
            entity_pair_map_file = arg
        elif opt in ("-o", "--out-file"):
            out_file = arg

    print 'Reading in data from ' + type_map_file
    entity_type_map = dict((entity, types.split(',')) for entity, types in (line.strip().split('\t') for line in open(type_map_file, 'r')))

    print 'Reading in data from ' + entity_map_file
    entity_idx_map = dict((entity, idx) for entity, idx in (line.strip().split('\t') for line in open(entity_map_file, 'r')))

    print 'Reading in data from ' + entity_pair_map_file
    entity_pair_idx_map = dict((e1+'\t'+e2, idx) for e1, e2, idx in (line.strip().split('\t') for line in open(entity_pair_map_file, 'r')))

    print 'Populating type entity map'
    type_entity_map = defaultdict(set)
    for e1, types in entity_type_map.iteritems():
        for type in types:
            type_entity_map[type].add(e1)

    # print 'Populating entity replacement map'
    # type_replacement_map = defaultdict(set)
    # for e1, types in entity_type_map.iteritems():
    #     e1_idx = entity_idx_map[e1]
    #     for t in types:
    #         for e2 in type_entity_map[t]:
    #             if e1 + '\t' + e2 in entity_pair_idx_map and e1 != e2:
    #                 e2_idx = entity_idx_map[e2]
    #                 type_replacement_map[e1].add(e2)
    #                 # type_replacement_map[e1_idx].add(e2_idx)

    print 'Populating entity pair replacement map'
    def add_ep_replacement(e1, e2, e2_replacement, entity_pair_idx_map, ep, type_replacement_map):
        ep_replacement = e1 + '\t' + e2_replacement
        if ep_replacement in entity_pair_idx_map and e1 != e2 and e2 != e2_replacement:
            ep_idx = entity_pair_idx_map[ep]
            ep_replacement_idx = entity_pair_idx_map[ep_replacement]
            type_replacement_map[ep_idx].add(ep_replacement_idx)
            # type_replacement_map[ep].add(ep_replacement)


    type_replacement_map = defaultdict(set)
    not_matched = 0
    for i, (ep, ep_idx) in enumerate(entity_pair_idx_map.iteritems()):
        e1, e2 = ep.split('\t')
        if e2 in entity_type_map:
            [add_ep_replacement(e1, e2, e2_replacement, entity_pair_idx_map, ep, type_replacement_map)
                for t in entity_type_map[e2] for e2_replacement in type_entity_map[t]]
        else:
            not_matched += 1
        if i % 100 == 0:
            sys.stdout.write('\r' + str(i) + '\t' + str(not_matched))
            sys.stdout.flush()

    def export_map(e1, e2_list, o):
        sys.stdout.flush()
        o.write(e1 + '\t')
        o.write(','.join(e2_list) + '\n')

    print 'exporting entity replacement map'
    out = gzip.open(out_file, 'wb')
    [export_map(entity1, entity2_list, out) for entity1, entity2_list in type_replacement_map.iteritems()]
    out.close()
    print 'Done'



if __name__ == "__main__":
    main(sys.argv[1:])
