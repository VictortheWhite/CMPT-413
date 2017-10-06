#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import queue
import argparse

# arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--unigramcounts", dest='counts1w', type=list, nargs='+', default=[os.path.join('data', 'count_1w.txt')], help="unigram counts")
argparser.add_argument("-b", "--bigramcounts", dest='counts2w', type=list, nargs='+', default=[os.path.join('data', 'count_2w.txt')], help="bigram counts")
argparser.add_argument("-i", "--inputfile", dest="input", type=str, default=os.path.join('data', 'input'), help="input file to segment")
argparser.add_argument("-m", "--maxlen", dest ='maxlen', type=int, default=10, help="max possible length for each word")
argparser.add_argument("-s", "--smooth", dest ='smooth', type=float, default=1.0, help="smoothing parameter")
argparser.add_argument("-l", "--log", dest='enable_log', type=bool, default=False, help="the flag that enable the log print")
args = argparser.parse_args()

class ProbDist(dict):
    """A probability distribution estimated from counts in datafile."""
    def __init__(self, filenames, sep='\t', totalvalue=None, smoothingfn=None):
        for filename in filenames:
            for line in open(filename, 'r'):
                (key, freq) = line.split(sep)
                self[key] = self.get(key, 0) + int(freq)
        self.totalvalue = float(totalvalue or sum(self.values()))
        self.totaltype = float(len(self))
        self.smoothingfn = smoothingfn or (lambda prob, v, t: (prob + args.smooth) / (v + args.smooth * t))

    def __call__(self, key):
        """Get probability for this key"""
        prob = self.get(key, 0) # if missing, the prob will be 0 before smoothing
        if prob == 0 and len(key) > 1:
            return 1e-200
        else:
            return self.smoothingfn(float(prob), self.totalvalue, self.totaltype)

    def log_prob(self, key):
        """Return log probability for this key"""
        return math.log(self(key))

# the default segmenter does not use any probabilities, but you could ...
prob_dist = ProbDist(args.counts1w)

# handle each line in input
with open(args.input) as f:
    for line in f:
        line = line.strip()

        # init input words list
        input_words = [i for i in line]
        if args.enable_log:
            print("==> Input words: ", input_words, len(input_words))

        # the dynamic programming table to store the argmax for every prefix of input
        chart = {}
        chart[-1] = ("", 1.0) # initial state: empty string with probability 1.0

        for i in range(len(input_words)):
            for j in range(1, args.maxlen + 1):
                if i - j + 1 < 0:
                    continue
                word = "".join(input_words[i-j+1:i+1])
                prob = prob_dist.log_prob(word)
                _, prev_prob = chart[i - j]
                if args.enable_log:
                    print("==> Check: ", i, j, word, prob, prev_prob)
                if i not in chart:
                    chart[i] = (word, prev_prob + prob)
                    if args.enable_log:
                        print("==> New: ", i, word, prev_prob + prob)
                else:
                    _, best_prob = chart[i]
                    if prev_prob + prob > best_prob:
                        chart[i] = (word, prev_prob + prob)
                        if args.enable_log:
                            print("==> Update: ", i, word, prev_prob + prob)

        # get the best segmentation
        index = len(input_words) - 1
        segmentation = []
        while index >= 0:
            segmented_word, _ = chart[index]
            segmentation.append(segmented_word)
            index -= len(segmented_word)
        segmentation.reverse()
        print(" ".join(segmentation))
