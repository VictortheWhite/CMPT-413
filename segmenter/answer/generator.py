#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

# arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input", dest='input', type=str, default=os.path.join('data', 'wseg_simplified_cn.txt'), help="input file")
argparser.add_argument("-o", "--output", dest='output', type=str, default=os.path.join('data', 'count_extra.txt'), help="output file")
args = argparser.parse_args()

dict = {}

for line in open(args.input, 'r'):
    words = line.split()
    for word in words:
        word = word.strip()
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1

output_file = open(args.output, 'w')
for word, count in dict.items():
    output_file.write('%s\t%d\n' % (word, count))
output_file.close()

