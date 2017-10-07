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
argparser.add_argument("--maxlen", dest ='maxlen', type=int, default=10, help="max possible length for unknown word")
argparser.add_argument("--smooth", dest ='smooth', type=float, default=0.0245, help="smoothing parameter")
argparser.add_argument("--bigram", dest ='enable_bigram', action='store_true', default=False, help="the flag that enable the bigram method")
argparser.add_argument("--log", dest='enable_log', action='store_true', default=False, help="the flag that enable the log print")
args = argparser.parse_args()


class ProbDist(dict):
    """A probability distribution estimated from counts in datafile."""
    def __init__(self, filenames, sep='\t'):
        for filename in filenames:
            for line in open(filename, 'r'):
                (key, freq) = line.split(sep)
                self[key] = self.get(key, 0) + int(freq)
        self.totalvalue = float(sum(self.values()))
        self.totaltype = float(len(self))

    def __call__(self, key):
        """Get probability for this key"""
        if key not in self:
            return self._smoothing_func(key)
        else:
            return float(self[key]) / self.totalvalue

    def _smoothing_func(self, key):
        """Better smoothing function"""
        if len(key) <= 1:
            return 1. / self.totalvalue
        else:
            score = 1. / self.totalvalue
            for i in range(1, len(key)):
                score = score / (args.smooth * i * self.totalvalue)
            return score + 1e-200

    def count(self, key):
        """Get count number for this key"""
        return self.get(key, 0)


class Unigram(object):
    """Unigram method to segment the sentense"""
    def __init__(self, input_words, prob_dist):
        self.input_words = input_words
        self.prob_dist = prob_dist
        self.chart = {} # the dynamic programming table to store the argmax for every prefix of input

    def segment(self):
        for i in range(len(self.input_words)):
            for j in range(1, args.maxlen + 1):
                if i - j + 1 < 0:
                    continue
                word = "".join(self.input_words[i-j+1:i+1])
                prob = math.log(self.prob_dist(word))
                prev_prob = self.chart[i - j][1] if i - j >=0 else 0
                if args.enable_log:
                    print("==> Check: ", i, j, word, prob, prev_prob)
                if i not in self.chart or (prev_prob + prob) > self.chart[i][1]:
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

    def get_probability(self, word1, word2):
        word_pair = word1 + ' ' + word2
        if word_pair in self.prob_dist2 and word1 in self.prob_dist:
            # Laplacian bigram probabilities
            return math.log(self.prob_dist2.count(word_pair) / self.prob_dist.count(word1))
        else:
            # Backoff
            return math.log(self.prob_dist(word2))

    def segment(self):
        for i in range(args.maxlen):
            word1 = "<S>"
            word2 = "".join(self.input_words[:i+1])
            prob = self.get_probability(word1, word2)
            self.chart[(i, len(word2))] = (word2, prob)

        for i in range(1, len(self.input_words)):
            for j in range(1, args.maxlen + 1):
                if i - j + 1 < 0:
                    continue
                for k in range(1, args.maxlen + 1):
                    if i - j - k + 1 < 0:
                        continue
                    word1 = "".join(self.input_words[i-j-k+1:i-j+1])
                    word2 = "".join(self.input_words[i-j+1:i+1])
                    prob = self.get_probability(word1, word2)
                    prev_prob = self.chart[(i - j, k)][1]
                    if args.enable_log:
                        print("==> Check: ", i - j, k, word1, word2, prob, prev_prob)
                    if (i, j) not in self.chart or (prev_prob + prob) > self.chart[(i, j)][1]:
                        self.chart[(i, j)] = (word2, prev_prob + prob)
                        if args.enable_log:
                            print("==> Update: ", i, j, word2, prev_prob + prob)

        # get the best segmentation
        best_chart = {}
        for (i, j) in self.chart.keys():
            if i not in best_chart or best_chart[i][1] < self.chart[(i, j)][1]:
                best_chart[i] = self.chart[(i, j)]

        index = len(self.input_words) - 1
        result = []
        while index >= 0:
            word, _ = best_chart[index]
            result.append(word)
            index -= len(word)
        result.reverse()
        print(" ".join(result))


if __name__ == "__main__":
    # the default segmenter does not use any probabilities, but you could ...
    prob_dist = ProbDist(args.counts1w)
    prob_dist2 = ProbDist(args.counts2w)

    # handle each line in input
    with open(args.input) as f:
        for line in f:
            line = line.strip()

            # init input words list
            input_words = [i for i in line]
            if args.enable_log:
                print("==> Input words: ", input_words, len(input_words))

            if args.enable_bigram:
                # the bigram method
                bigram = Bigram(input_words, prob_dist, prob_dist2)
                bigram.segment()
            else:
                # the unigram method
                unigram = Unigram(input_words, prob_dist)
                unigram.segment()
