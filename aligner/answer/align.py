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
argparser.add_argument("-s", "--smoothing", dest="smooth", default=0.001, type=float, help="add_n smoothing value")
argparser.add_argument("-S", "--Smoothing", dest="Smooth", default=0.003, type=float, help="add_n smoothing value for distortion")
argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=2**64, type=int, help="Number of sentences to use for training and alignment")
argparser.add_argument("-i", "--num_iteration", dest="num_iter", default=5, type=int, help="Number of iteration/epoch number")
args = argparser.parse_args()

def align(bitext):
    t = defaultdict(float)
    a = defaultdict(float)
    
    sys.stderr.write("Init parameters...\n")
    count_e = defaultdict(float)
    count_f = defaultdict(float)
    count_ef = defaultdict(float)
    for f, e in bitext:
        I = len(f)
        J = len(e)
        for i, f_i in enumerate(f):
            for j, e_j in enumerate(e):
                count_f[f_i] += 1.0
                count_e[e_j] += 1.0
                count_ef[(f_i, e_j)] += 1.0
                a[(j, i, I, J)] = 1.0 / (J+1)
            a[-1, i, I, J] = 1.0 / (J+1)
        #t[f_i, None] = 1.0 / (I * J)
    for f, e in count_ef:
        t[(f, e)] = count_ef[(f, e)] / (count_e[e] * count_f[f])

    V, S = 100000, 10000
    # traning
    for T in range(args.num_iter):
        sys.stderr.write("Starting Iteration %d ...\n" % T)
        expected_count_t_fe = defaultdict(float)
        expected_count_t_e = defaultdict(float)
        expected_count_a_fe = defaultdict(float)
        expected_count_a_e = defaultdict(float)

        for k, (f, e) in enumerate(bitext):
            if k % 10000 == 0:
                 sys.stderr.write("     %d\n" % k)
            I, J = len(f), len(e)
            for i in range(I):
                # calculate the normalization term
                z = t[(f[i], None)] * a[(-1, i, I, J)]
                for j in range(J):
                    z += t[(f[i], e[j])] * a[(j, i, I, J)]
                # calculate expected count
                for j in range(J):
                    c = t[(f[i], e[j])] * a[(j, i, I, J)] / z
                    expected_count_t_fe[(f[i], e[j])] += c
                    expected_count_t_e[e[j]] += c
                    expected_count_a_fe[(j, i, I, J)] += c
                    expected_count_a_e[(i, I, J)] += c

                c = t[(f[i], None)] * a[(-1, i, I, J)] / z
                expected_count_t_fe[(f_i, None)] += c
                expected_count_t_e[None] += c
                expected_count_a_fe[(-1, i, I, J)] += c
                expected_count_a_e[(i, I, J)] += c
        
        # smoothing
        sys.stderr.write("smoothing...\n")
        for f, e in expected_count_t_fe:
            t[(f, e)] = (expected_count_t_fe[(f, e)] + args.smooth) / (expected_count_t_e[e] + args.smooth * V)
        for i, j, I, J in expected_count_a_fe:
            a[(i, j, I, J)] = (expected_count_a_fe[(i, j, I, J)] + args.Smooth) / (expected_count_a_e[j, I, J] + args.Smooth * S)
    
    # align and output result
    sys.stderr.write("aligning...\n")
    alignments = []
    for f, e in bitext:
        I, J = len(f), len(e)
        alignments.append([])
        for i, f_i in enumerate(f):
            best_p = t[(f_i, None)] * a[-1, i, I, J]
            best_j = -1
            for j, e_j in enumerate(e):
                p = t[(f_i, e_j)] * a[j, i, I, J]
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
    bitext = [[sentence.strip().split() for sentence in pair] for pair in list(zip(open(f_data), open(e_data)))[:args.num_sents]]
    bitext_2 = [[sentence.strip().split() for sentence in pair] for pair in list(zip(open(e_data), open(f_data)))[:args.num_sents]]

    aligns_1 = align(bitext)
    aligns_2 = align(bitext_2)

    # output intersection of 2 alignments
    sys.stderr.write("intersecting aligns...\n")
    for k, a in enumerate(aligns_1):
        for i, j in a:
            if (j, i) in aligns_2[k]:
                sys.stdout.write("%i-%i " % (i, j))
        sys.stdout.write("\n")
        
    