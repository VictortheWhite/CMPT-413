#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

from collections import defaultdict

argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
argparser.add_argument("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
argparser.add_argument("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
argparser.add_argument("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
argparser.add_argument("-t", "--smoothing-t", dest="smooth_t", default=0.001, type=float, help="add_n smoothing value for t")
argparser.add_argument("-q", "--smoothing-q", dest="smooth_q", default=0.0004, type=float, help="add_n smoothing value for q(distortion)")
argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=2**64, type=int, help="Number of sentences to use for training and alignment")
argparser.add_argument("-i", "--num_iteration", dest="num_iter", default=5, type=int, help="Number of iteration/epoch number")
args = argparser.parse_args()

def align(bitext):
    sys.stderr.write("Init parameters...\n")

    count_e = defaultdict(float)
    count_f = defaultdict(float)
    count_fe = defaultdict(float)

    q = defaultdict(lambda: 1.0/15, {})
    for f, e in bitext:
        l, m = len(e), len(f)
        for i, f_i in enumerate(f):
            for j, e_j in enumerate(e):
                count_f[f_i] += 1
                count_e[e_j] += 1
                count_fe[(f_i, e_j)] += 1
                q[(j, i, l, m)] = 1.0 / (l + 1)
            count_e[None] += 1
            count_fe[(f_i, None)] += 1
            q[(None, i, l, m)] = 1.0 / (l + 1)

    t = defaultdict(lambda: 1.0/len(count_f), {})
    for f, e in count_fe:
        t[(f, e)] = count_fe[(f, e)] / (count_e[e] * count_f[f])

    V_t, V_q = 100000, 10000

    # traning
    for k in range(args.num_iter):
        sys.stderr.write("Starting Iteration %d ...\n" % k)
        expected_count_t_fe = defaultdict(float)
        expected_count_t_e = defaultdict(float)
        expected_count_q_fe = defaultdict(float)
        expected_count_q_e = defaultdict(float)

        for f, e in bitext:
            l, m = len(e), len(f)
            for i, f_i in enumerate(f):
                # calculate the normalization term
                z = t[(f_i, None)] * q[(None, i, l, m)]
                for j, e_j in enumerate(e):
                    z += t[(f_i, e_j)] * q[(j, i, l, m)]
                # calculate expected count
                for j, e_j in enumerate(e):
                    c = t[(f_i, e_j)] * q[(j, i, l, m)] / z
                    expected_count_t_fe[(f_i, e_j)] += c
                    expected_count_t_e[e_j] += c
                    expected_count_q_fe[(j, i, l, m)] += c
                    expected_count_q_e[(i, l, m)] += c

                c = t[(f_i, None)] * q[(None, i, l, m)] / z
                expected_count_t_fe[(f_i, None)] += c
                expected_count_t_e[None] += c
                expected_count_q_fe[(None, i, l, m)] += c
                expected_count_q_e[(i, l, m)] += c
        
        # smoothing
        for f, e in expected_count_t_fe:
            t[(f, e)] = (expected_count_t_fe[(f, e)] + args.smooth_t) / (expected_count_t_e[e] + args.smooth_t * V_t)
        for j, i, l, m in expected_count_q_fe:
            q[(j, i, l, m)] = (expected_count_q_fe[(j, i, l, m)] + args.smooth_q) / (expected_count_q_e[i, l, m] + args.smooth_q * V_q)
    
    # align and output result
    sys.stderr.write("Save alignment...\n")
    alignments = []
    for f, e in bitext:
        l, m = len(e), len(f)
        alignments.append([])
        for i, f_i in enumerate(f):
            best_p = t[(f_i, None)] * q[None, i, l, m]
            best_j = -1
            for j, e_j in enumerate(e):
                p = t[(f_i, e_j)] * q[j, i, l, m]
                if p > best_p:
                    best_p = p
                    best_j = j
            if best_j >= 0:
                alignments[-1].append((i, best_j))

    return alignments

if __name__ == '__main__':
    sys.stderr.write("Read data...\n")
    f_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.french)
    e_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.english)
    bitext_1 = [[sentence.strip().split() for sentence in pair] for pair in list(zip(open(f_data, encoding='utf8'), open(e_data, encoding='utf8')))[:args.num_sents]]
    bitext_2 = [[sentence.strip().split() for sentence in pair] for pair in list(zip(open(e_data, encoding='utf8'), open(f_data, encoding='utf8')))[:args.num_sents]]

    aligns_1 = align(bitext_1)
    aligns_2 = align(bitext_2)

    # output 2 intersection of alignments
    for k, a in enumerate(aligns_1):
        for i, j in a:
            if (j, i) in aligns_2[k]:
                sys.stdout.write("%i-%i " % (i, j))
        sys.stdout.write("\n")