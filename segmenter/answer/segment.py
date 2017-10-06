#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import queue
import argparse

# arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--unigramcounts", dest='counts1w', type=list, nargs='+', default=[os.path.join('data', 'count_1w.txt')], help="unigram counts")
argparser.add_argument("--bigramcounts", dest='counts2w', type=list, nargs='+', default=[os.path.join('data', 'count_2w.txt')], help="bigram counts")
argparser.add_argument("--inputfile", dest="input", type=str, default=os.path.join('data', 'input'), help="input file to segment")
argparser.add_argument("--maxlen", dest ='maxlen', type=int, default=10, help="max possible length for each word")
argparser.add_argument("--smooth", dest ='smooth', type=float, default=1.0, help="smoothing parameter")
argparser.add_argument("--bigram", dest ='enable_bigram', type=bool, default=True, help="the flag that enable the bigram method")
argparser.add_argument("--log", dest='enable_log', type=bool, default=False, help="the flag that enable the log print")
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


class Unigram(object):
    """Unigram method to segment the sentense"""
    def __init__(self, input_words, prob_dist):
        self.input_words = input_words
        self.prob_dist = prob_dist
        self.chart = {} # the dynamic programming table to store the argmax for every prefix of input
        self.chart[-1] = ("", 1.0) # initial state: empty string with probability 1.0

    def segment(self):
        for i in range(len(self.input_words)):
            for j in range(1, args.maxlen + 1):
                if i - j + 1 < 0:
                    continue
                word = "".join(self.input_words[i-j+1:i+1])
                prob = self.prob_dist.log_prob(word)
                _, prev_prob = self.chart[i - j]
                if args.enable_log:
                    print("==> Check: ", i, j, word, prob, prev_prob)
                if i not in self.chart:
                    self.chart[i] = (word, prev_prob + prob)
                    if args.enable_log:
                        print("==> New: ", i, word, prev_prob + prob)
                else:
                    _, best_prob = self.chart[i]
                    if prev_prob + prob > best_prob:
                        self.chart[i] = (word, prev_prob + prob)
                        if args.enable_log:
                            print("==> Update: ", i, word, prev_prob + prob)

        # get the best segmentation
        index = len(self.input_words) - 1
        result = []
        while index >= 0:
            word, _ = self.chart[index]
            result.append(word)
            index -= len(word)
        result.reverse()
        print(" ".join(result))


class Bigram(object):
    """Bigram method to segment the sentense"""
    def __init__(self, input_words, prob_dist, prob_dist2):
        self.input_words = input_words
        self.prob_dist = prob_dist
        self.prob_dist2 = prob_dist2
        self.chart = {} # the dynamic programming table to store the argmax for every prefix of input
        self.chart[-1] = ("", 1.0) # initial state: empty string with probability 1.0

    def segment(self):
        for i in range(len(self.input_words)):
            for j in range(1, args.maxlen + 1):
                if i - j + 1 < 0:
                    continue
                word = "".join(self.input_words[i-j+1:i+1])
                prob = self.prob_dist.log_prob(word)
                _, prev_prob = self.chart[i - j]
                if args.enable_log:
                    print("==> Check: ", i, j, word, prob, prev_prob)
                if i not in self.chart:
                    self.chart[i] = (word, prev_prob + prob)
                    if args.enable_log:
                        print("==> New: ", i, word, prev_prob + prob)
                else:
                    _, best_prob = self.chart[i]
                    if prev_prob + prob > best_prob:
                        self.chart[i] = (word, prev_prob + prob)
                        if args.enable_log:
                            print("==> Update: ", i, word, prev_prob + prob)

        # get the best segmentation
        index = len(self.input_words) - 1
        result = []
        while index >= 0:
            word, _ = self.chart[index]
            result.append(word)
            index -= len(word)
        result.reverse()
        print(" ".join(result))


# the default segmenter does not use any probabilities, but you could ...
prob_dist = ProbDist(args.counts1w)
prob_dist2 = {}

# handle each line in input
with open(args.input) as f:
    for line in f:
        line = line.strip()

        # init input words list
        input_words = [i for i in line]
        if args.enable_log:
            print("==> Input words: ", input_words, len(input_words))

        if not args.enable_bigram:
            # the unigram method
            unigram = Unigram(input_words, prob_dist)
            unigram.segment()
        else:
            # the bigram method
            bigram = Bigram(input_words, prob_dist, prob_dist2)
            bigram.segment()
