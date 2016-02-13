import sys
import getopt
import networkx as nx


####
#   finds and exports the largest connected component using a 4 col mtx file
#   use '-a' flag if data already int-mapped by $TH_RELEX_ROOT/bin/process/StringFile2IntFile,
####


def main(argv):
    in_file = ''
    out_file = ''
    min_degree_prune = 2
    already_int_mapped = False

    help_msg = 'LargestConnectedComponent.py -i <inFile> -o <outputfile> -d <minimum degree> -a [already int mapped]'
    try:
        opts, args = getopt.getopt(argv, "hi:o:d:a", ["inFile=", "outFile="])
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
        elif opt in ("-d", "--degree"):
            min_degree_prune = int(arg)
        elif opt in ("-a", "--already-int-mapped"):
            already_int_mapped = True

    ep_map = {}
    rel_map = {}

    def process_line(line, g):
        if already_int_mapped:
            e1, e2, ep, rel, tokens, label = line.strip().split('\t')
            g.add_edge(int(ep), int(rel) * -1)
        else:
            e1_str, e2_str, rel_str, label = line.strip().split('\t')
            ep_str = e1_str + '\t' + e2_str
            ep_map.setdefault(ep_str, len(ep_map) + 1)
            rel_map.setdefault(rel_str, (len(rel_map) + 1) * -1)
            g.add_edge(ep_map[ep_str], rel_map[rel_str])

    print 'Adding edges to graph'
    graph = nx.Graph()
    [process_line(cur_line, graph) for cur_line in open(in_file, 'r')]
    print('Initial graph contains ' + str(len(graph.nodes())) + ' nodes, '
          + str(len(graph.edges())) + ' edges')

    print 'Pruning nodes with degree < ' + str(min_degree_prune)
    remove = [node for node, degree in graph.degree().items() if degree < min_degree_prune]
    graph.remove_nodes_from(remove)
    print 'Pruned ' + str(len(remove)) + ' nodes'

    print 'Finding largest connected component'
    largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    print('Largest component contains ' + str(len(largest_component.nodes())) + ' nodes, '
          + str(len(largest_component.edges())) + ' edges')

    def export_line(line, f_out):
        if already_int_mapped:
            e1, e2, ep, rel, tokens, label = line.strip().split('\t')
            if largest_component.has_edge(int(ep), int(rel) * -1):
                f_out.write(line)
        else:
            e1_str, e2_str, rel_str, label = line.strip().split('\t')
            ep_str = e1_str + '\t' + e2_str
            if largest_component.has_edge(ep_map[ep_str], rel_map[rel_str]):
                f_out.write(line)

    # we need to reconstruct the data from the original file -
    # to use less memory we didn't store the data but will iterate over it again
    print 'Exporting largest component to ' + out_file
    with open(out_file, 'w') as out:
        [export_line(cur_line, out) for cur_line in open(in_file, 'r')]

    print 'Done'


if __name__ == "__main__":
    main(sys.argv[1:])
