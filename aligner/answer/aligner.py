#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

from collections import defaultdict

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    argparser.add_argument("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    argparser.add_argument("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    argparser.add_argument("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    argparser.add_argument("-s", "--smoothing", dest="smooth", default=0.01, type=float, help="add_n smoothing value")
    argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=2**64, type=int, help="Number of sentences to use for training and alignment")
    argparser.add_argument("-i", "--num_iteration", dest="num_iter", default=5, type=int, help="Number of iteration/epoch number")
    args = argparser.parse_args()

    sys.stderr.write("Read data...\n")
    f_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.french)
    e_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.english)
    bitext = [[sentence.strip().split() for sentence in pair] for pair in list(zip(open(f_data), open(e_data)))[:args.num_sents]]

    sys.stderr.write("Init parameters...\n")

    # calculate V (length of f words)
    f_count = defaultdict(int)
    for f, e in bitext:
        for f_i in set(f):
            f_count[f_i] += 1
    V = len(f_count)

    # init t uniformly
    t = defaultdict(float)
    for f, e in bitext:
        for f_i in set(f):
            for e_j in set(e).union({None}):
                t[(f_i, e_j)] = 1.0 / V

    # traning
    for i in range(args.num_iter):
        sys.stderr.write("Starting Iteration %d ...\n" % i)
        expected_count_fe = defaultdict(float)
        expected_count_e = defaultdict(float)

        for f, e in bitext:
            for f_i in f:
                # calculate the normalization term
                z = t[(f_i, None)]
                for e_j in e:
                    z += t[(f_i, e_j)]
                # calculate expected count
                for e_j in e:
                    c = t[(f_i, e_j)] / z
                    expected_count_fe[(f_i, e_j)] += c
                    expected_count_e[e_j] += c
                c = t[(f_i, None)] / z
                expected_count_fe[(f_i, None)] += c
                expected_count_e[None] += c

        for f, e in expected_count_fe:
            t[(f, e)] = (expected_count_fe[(f, e)] + args.smooth) / (expected_count_e[e] + args.smooth * V)

    # align and output result
    for f, e in bitext:
        for i, f_i in enumerate(f):
            best_p = t[(f_i, None)]
            best_j = -1
            for j, e_j in enumerate(e):
                p = t[(f_i, e_j)]
                if p > best_p:
                    best_p = p
                    best_j = j
            if best_j >= 0:
                sys.stdout.write("%i-%i " % (i, best_j))
        sys.stdout.write("\n")