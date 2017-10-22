#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import perc
import argparse
from collections import defaultdict

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def calculate_feat(chunk_tags, feat_list):
    feat_vec = defaultdict(int)
    index = 0
    for i in range(2, len(chunk_tags)-2):
        (index, feats) = perc.feats_for_word(index, feat_list)
        if len(feats) == 0:
            raise ValueError("Returned empty feature")
        for feat in feats:
            if feat == 'B':
                feat_vec['B:' + chunk_tags[i - 1], chunk_tags[i]] += 1
            else:
                feat_vec[feat, chunk_tags[i]] += 1
    return feat_vec

def perc_train(train_data, tagset, numepochs):
    feat_vec = defaultdict(int)
    feat_vec_sum = defaultdict(int)
    sentence_count = total_sentence_count = numepochs * len(train_data)

    for epoch in range(numepochs):
        print("Epoch %d..." % epoch)

        for labeled_list, feat_list in train_data:
            default_tag = tagset[0]

            true_chunk_tags = list(map(lambda label: label.split()[-1], labeled_list))
            true_chunk_tags.insert(0, '_B-1')
            true_chunk_tags.insert(0, '_B-2')
            true_chunk_tags.append('_B+1')
            true_chunk_tags.append('_B+2')
            true_chunk_tags = calculate_feat(true_chunk_tags, feat_list)

            output_chunk_tags = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)
            output_chunk_tags.insert(0, '_B-1')
            output_chunk_tags.insert(0, '_B-2')
            output_chunk_tags.append('_B+1')
            output_chunk_tags.append('_B+2')
            output_chunk_tags = calculate_feat(output_chunk_tags, feat_list)

            if true_chunk_tags != output_chunk_tags:
                # plus correct weight
                for key in true_chunk_tags:
                    value = feat_vec.get(key, 0) + true_chunk_tags[key]
                    if value != 0:
                        feat_vec[key] = value
                    else:
                        del feat_vec[key]
                    value = feat_vec_sum.get(key, 0) + true_chunk_tags[key] * sentence_count / total_sentence_count
                    if value != 0:
                        feat_vec_sum[key] = value
                    else:
                        del feat_vec_sum[key]

                # minus incorrect weight
                for key in output_chunk_tags:
                    value = feat_vec.get(key, 0) - output_chunk_tags[key]
                    if value != 0:
                        feat_vec[key] = value
                    else:
                        del feat_vec[key]
                    value = feat_vec_sum.get(key, 0) - output_chunk_tags[key] * sentence_count / total_sentence_count
                    if value != 0:
                        feat_vec_sum[key] = value
                    else:
                        del feat_vec_sum[key]

            sentence_count -= 1

    return feat_vec_sum

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--tagsetfile", dest="tagsetfile", type=str, default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    argparser.add_argument("-i", "--trainfile", dest="trainfile", type=str, default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    argparser.add_argument("-f", "--featfile", dest="featfile", type=str, default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    argparser.add_argument("-e", "--numepochs", dest="numepochs", type=int, default=10, help="number of epochs of training; in each epoch we iterate over over all the training examples")
    argparser.add_argument("-m", "--modelfile", dest="modelfile", type=str, default=os.path.join("model", "model"), help="weights for all features stored on disk")
    args = argparser.parse_args()

    # each element in the feat_vec dictionary is: key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    eprint("👀  reading data ...")
    tagset = perc.read_tagset(args.tagsetfile)
    train_data = perc.read_labeled_data(args.trainfile, args.featfile)
    eprint("✨  done.")

    eprint("🤖  training ...")
    feat_vec = perc_train(train_data, tagset, args.numepochs)
    eprint("✨  done.")

    eprint("💾  saving model ...")
    perc.perc_write_to_file(feat_vec, args.modelfile)
    eprint("✨  done.")
