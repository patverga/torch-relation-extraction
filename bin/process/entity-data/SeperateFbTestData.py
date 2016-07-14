###
# Takes a 2 col tsv 'fb_id \t fb_type'
#  Randomly generates train, valid, test split
#  exports to format required by process data
###


import os
import sys
import getopt
from collections import defaultdict
import random
from random import shuffle

def main(argv):
    in_file = ''
    out_dir = ''
    train_portion = 1
    valid_portion = 0
    test_portion = 0
    # number of random entities to hold out of training set
    leave_out = 0
    negatives = -1
    prefix = ''
    random.seed(0)

    help_msg = 'SeperateFbTestData.py.py -i <inFile> -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:t:v:n:p:l:", ["inFile=", "outDir=", "train=", "valid="])
    except getopt.GetoptError:
        print help_msg
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit()
        elif opt in ("-i", "--inFile"):
            in_file = arg
        elif opt in ("-o", "--outDir"):
            out_dir = arg
        elif opt in ("-t", "--test"):
            test_portion = float(arg)
        elif opt in ("-v", "--valid"):
            valid_portion = float(arg)
        elif opt in ("-n", "--negatives"):
            negatives = int(arg)
        elif opt in ("-p", "--prefix"):
            prefix = arg + '\t'
        elif opt in ("-l", "--leaveOut"):
            leave_out = int(arg)

    train_portion = train_portion - valid_portion - test_portion


    # make output dir
    if not os.path.exists(out_dir + '/valid_seen'):
        os.makedirs(out_dir + '/valid_seen')
        os.makedirs(out_dir + '/valid_unseen')
        os.makedirs(out_dir + '/test_seen')
        os.makedirs(out_dir + '/test_unseen')

    type_entity_dict = defaultdict(set)
    train_type_entity_dict = defaultdict(set)
    seen_valid_type_entity_dict = defaultdict(set)
    unseen_valid_type_entity_dict = defaultdict(set)
    seen_test_type_entity_dict = defaultdict(set)
    unseen_test_type_entity_dict = defaultdict(set)
    entity_list = []

    print 'Loading lines from ' + in_file + ' to map'
    for line in open(in_file, 'r'):
        fb_id, fb_type = line.strip().split('\t')
        type_entity_dict[fb_type].add(fb_id)
        entity_list.append(fb_id)

    # select subset of entities to leave out of training and use for testing
    if leave_out > 0:
        shuffle(entity_list)
        held_out_entities = set(entity_list[:leave_out])
        with open(out_dir+'/heldout-entities.txt', 'w') as out:
            for e in entity_list[:leave_out]:
                out.write(e+'\n')
        entity_list = entity_list[0:leave_out]

    print 'Splitting train valid test'
    entities_in_train = set()
    for fb_type, entities in type_entity_dict.iteritems():
        x = [i for i in entities]
        shuffle(x)
        # make sure each entity appears in train atleast once
        x_train = set([i for i in x if i not in entities_in_train])
        entities_in_train |= x_train

        train_len = int(len(x)*train_portion)
        valid_len = int(len(x)*valid_portion)
        test_len = int(len(x)*test_portion)
        # add remainder to train
        train_len += len(x) - (train_len + valid_len + test_len)

        train = x[:train_len] + [i for i in x_train]
        valid = [i for i in x[train_len:train_len+valid_len] if i not in x_train]
        test = [i for i in x[train_len+valid_len:train_len+valid_len+test_len] if i not in x_train]

        # split test set into entities seen during training and those unseen
        unseen_valid = [e for e in valid if e in held_out_entities]
        unseen_test = [e for e in test if e in held_out_entities]
        seen_valid = [e for e in valid if e not in held_out_entities]
        seen_test = [e for e in test if e not in held_out_entities]

        train_type_entity_dict[fb_type] |= (set(train))
        unseen_valid_type_entity_dict[fb_type] |= (set(unseen_valid))
        seen_valid_type_entity_dict[fb_type] |= (set(seen_valid))
        unseen_test_type_entity_dict[fb_type] |= (set(unseen_test))
        seen_test_type_entity_dict[fb_type] |= (set(seen_test))

    print 'Exporting lines to ' + out_dir
    train_out = open(out_dir + '/fb_train.mtx', 'w')
    for fb_type, pos_entities in type_entity_dict.iteritems():
        # write to train
        [train_out.write(prefix+entity + '\t' + fb_type + '\t1\n') for entity in train_type_entity_dict[fb_type]]
        # write valid
        export_file(pos_entities, seen_valid_type_entity_dict[fb_type], entity_list, fb_type,
                    negatives, out_dir + '/valid_seen/' + fb_type.replace('/', '_'), prefix)
        export_file(pos_entities, unseen_valid_type_entity_dict[fb_type], entity_list, fb_type,
                    negatives, out_dir + '/valid_unseen/' + fb_type.replace('/', '_'), prefix)
        # write test
        export_file(pos_entities, seen_test_type_entity_dict[fb_type], entity_list, fb_type,
                    negatives, out_dir + '/test_seen/' + fb_type.replace('/', '_'), prefix)
        export_file(pos_entities, unseen_test_type_entity_dict[fb_type], entity_list, fb_type,
                    negatives, out_dir + '/test_unseen/' + fb_type.replace('/', '_'), prefix)

    train_out.close()

    print 'Done'


def export_file(pos_entities, pos_entity_subset, entity_list, fb_type, negatives, out_file, prefix):
    if len(pos_entity_subset) > 0:
        print 'Writing to ' + out_file
        with open(out_file, 'w') as out:
            # randomly select n negatives for each positive
            neg_indices = [i for i in range(len(entity_list)) if entity_list[i] not in pos_entities]
            shuffle(neg_indices)
            neg_indices = neg_indices if negatives < 0 else neg_indices[:(negatives * len(pos_entity_subset))]
            # write all positives to file
            [out.write(prefix + entity + '\t' + fb_type + '\t1\n') for entity in pos_entity_subset]
            # write all negatives to file
            [out.write(prefix + entity_list[i] + '\t' + fb_type + '\t0\n') for i in neg_indices]


if __name__ == "__main__":
    main(sys.argv[1:])
