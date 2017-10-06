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
        prob = self.get(key, 0) # if missing, the prob will be 0
        return self.smoothingfn(float(prob), self.totalvalue, self.totaltype)

    def log_prob(self, key):
        """Return log probability for this key"""
        return math.log(self(key))

class Entry(object):
    """Entry that stored in heap"""
    def __init__(self, word, start_position, log_prob, back_pointer):
        self.word = word
        self.start_position = start_position
        self.log_prob = log_prob
        self.back_pointer = back_pointer

    def __lt__(self, other):
        return self.start_position < other.start_position

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

        # init priority queue (heap)
        heap = queue.PriorityQueue()
        missing = True
        for word in prob_dist:
            if line.startswith(word):
                heap.put(Entry(word, 0, prob_dist.log_prob(word), None))
                missing = False
                if args.enable_log:
                    print("==> put entry: ", (word, 0, prob_dist.log_prob(word), None))
        if missing:
            new_entry = Entry(line[0], 0, prob_dist.log_prob(line[0]), None)
            heap.put(new_entry)
            if args.enable_log:
                print("==> put entry: ", (line[0], 0, prob_dist.log_prob(line[0]), None))

        # iteratively fill in chart[i] for all i
        while not heap.empty():
            entry = heap.get()
            if args.enable_log:
                print("==> get entry: ", (entry.word, entry.start_position, entry.log_prob, entry.back_pointer))
            end_position = entry.start_position + len(entry.word) - 1
            if end_position in chart and entry.log_prob <= chart[end_position].log_prob:
                continue
            chart[end_position] = entry

            next_start_position = end_position + 1
            if next_start_position != len(input_words):
                missing = True
                for word in prob_dist:
                    if line[next_start_position:].startswith(word):
                        new_entry = Entry(word, next_start_position, prob_dist.log_prob(word), entry)
                        heap.put(new_entry)
                        missing = False
                        if args.enable_log:
                            print("==> put entry: ", (word, next_start_position, prob_dist.log_prob(word), entry))
                if missing:
                    new_entry = Entry(line[next_start_position], next_start_position, prob_dist.log_prob(line[next_start_position]), entry)
                    heap.put(new_entry)
                    if args.enable_log:
                        print("==> put entry: ", (line[next_start_position], next_start_position, prob_dist.log_prob(line[next_start_position]), entry))

        if args.enable_log:
            print("==> Chart: ", chart)

        # get the best segmentation
        segmentation = []
        segmented_entry = chart[len(input_words) - 1]
        while segmented_entry is not None:
            segmentation.append(segmented_entry.word)
            segmented_entry = segmented_entry.back_pointer
        segmentation.reverse()
        print(" ".join(segmentation))
