"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair.

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""

import perc
import sys, optparse, os
import copy
from collections import defaultdict

def perc_train(train_data, tagset, numepochs):
    print len(train_data)
    feat_vec = defaultdict(int)
    defaultTag = tagset[0]
    for i in range(numepochs):
        print i
        k = 0
        feat_index = 0
        for (labeled_list, feat_list) in train_data:
            if k % 100 == 0: print "     ", k
            k += 1
            z = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, defaultTag)

            # get the augmented labels and feats for the word
            labels = copy.deepcopy(labeled_list)
            (feat_index, feats) = perc.feats_for_word(feat_index, feat_list)
            labels.insert(0, '_B-1 _B-1 _B-1')
            z.insert(0, '_B-1')

            # update weights when t != labels[j]
            N = len(labels)
            for j in range(1, N-1):
                if x(labels, j, 2) != z[j]:
                    updateWeights(feat_vec, labels, z, j, feats)
    return feat_vec

def updateWeights(feat_vec, labels, z, labelIndex, feats):
    # check all feats
    for feat in feats:
        feat_name = feat.split(':')[0]
        if feat_name == 'B':
            feat_vec_update(feat_vec, ("B:"+z[labelIndex-1], z[labelIndex]), -1)
            feat_vec_update(feat_vec, ("B:"+x(labels, labelIndex-1, 2), x(labels, labelIndex, 2)), 1)
        else:
            feat_vec_update(feat_vec, (feat, z[labelIndex]), -1)
            feat_vec_update(feat_vec, (feat, x(labels, labelIndex, 2)), 1)


def x(labels, row, col):
    return labels[row].split()[col]

def feat_vec_update(feat_vec, key, adder):
    # drop 0s to make the dict small
    val = feat_vec.get(key, 0) + adder
    if val == 0:
        del feat_vec[key]
    else:
        feat_vec[key] = val



if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    optparser.add_option("-d", "--devmode", dest="devmode", action="store_true", default=False, help="devmode uses dev dataset and test set")
    (opts, _) = optparser.parse_args()


    if opts.devmode:
        opts.trainfile = "data/train.dev"
        opts.featfile = "data/train.feats.dev"
        opts.modelfile = "model"
        opts.numepochs = 10

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    print feat_vec
    perc.perc_write_to_file(feat_vec, opts.modelfile)
