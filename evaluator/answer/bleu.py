#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import math
import nltk

from collections import defaultdict
from itertools import islice
from bleu_lib import sentence_bleu

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
                        help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
                        help='Number of hypothesis pairs to evaluate')
    args = parser.parse_args()

    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(args.input, encoding='utf-8') as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

    for h1, h2, ref in islice(sentences(), args.num_sentences):
        h1_score = sentence_bleu(h1, ref, weights=(0.35, 0.30, 0.20, 0.15))
        h2_score = sentence_bleu(h2, ref, weights=(0.35, 0.30, 0.20, 0.15))
        if h1_score > h2_score:
            print(1)
        elif h1_score == h2_score:
            print(0)
        else:
            print(-1)

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
