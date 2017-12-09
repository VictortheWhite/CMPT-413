#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

from collections import defaultdict

def main():
    file_weights = {
        'output-svm-0.546': 4,
        'output-wordnet-0.538': 3,
        'output-meteor-0.535': 2,
        'output-bleu-0.529': 1,
    }
    score = defaultdict(int)
    max_line = 0
    for file in file_weights:
        with open(file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                res = int(line.strip())
                score[str(i) + ' ' + str(res)] += file_weights[file]
                max_line = max(i, max_line)
    for i in range(max_line + 1):
        pos = score[str(i) + ' 1']
        neg = score[str(i) + ' -1']
        eq = score[str(i) + ' 0']
        if pos > neg and pos >= eq:
            print(1)
        elif neg > pos and neg >= eq:
            print(-1)
        else:
            print(0)

if __name__ == '__main__':
    main()
