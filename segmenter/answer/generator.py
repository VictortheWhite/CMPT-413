#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

# arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--input", dest='input', type=str, default=os.path.join('data', 'wseg_simplified_cn.txt'), help="input file")
argparser.add_argument("--output1w", dest='output1w', type=str, default=os.path.join('data', 'count_1w_extra.txt'), help="output file")
argparser.add_argument("--output2w", dest='output2w', type=str, default=os.path.join('data', 'count_2w_extra.txt'), help="output file")
args = argparser.parse_args()

dict1w = {}
dict2w = {}

for line in open(args.input, 'r'):
    words = line.split()
    words = [word.strip() for word in words]

    prev = '<S>'

    for word in words:
        if word in dict1w:
            dict1w[word] += 1
        else:
            dict1w[word] = 1

        if (prev, word) in dict2w:
            dict2w[(prev, word)] += 1
        else:
            dict2w[(prev, word)] = 1
        prev = word


output1w = open(args.output1w, 'w')
for word, count in dict1w.items():
    output1w.write('%s\t%d\n' % (word, count))
output1w.close()

output2w = open(args.output2w, 'w')
for (word1, word2), count in dict2w.items():
    output2w.write('%s %s\t%d\n' % (word1, word2, count))
output2w.close()
