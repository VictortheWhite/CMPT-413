#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import codecs
import argparse

# Arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--unigramcounts", dest='counts1w', type=str, default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
argparser.add_argument("-b", "--bigramcounts", dest='counts2w', type=str, default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
argparser.add_argument("-i", "--inputfile", dest="input", type=str, default=os.path.join('data', 'input'), help="input file to segment")
args = argparser.parse_args()

class ProbDist(dict):
    """A probability distribution estimated from counts in datafile."""
    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self:
            return float(self[key])/float(self.N)
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1:
            return self.missingfn(key, self.N)
        else:
            return None

# the default segmenter does not use any probabilities, but you could ...
prob_dist = ProbDist(args.counts1w)

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)

# ignoring the dictionary provided in args.counts
with open(args.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        output = [i for i in utf8line]  # segmentation is one word per character in the input
        print " ".join(output)

sys.stdout = old
