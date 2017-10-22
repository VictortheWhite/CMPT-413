#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
count sentences in file that have one word per line and two newlines between sentences,
the file can be gzipped but if so it must have the .gz filename suffix.

>>> python count.py -i data/input.feats.gz
>>> python count.py -i data/input.txt.gz
"""

import re
import sys
import gzip
import logging
import argparse

def countSentences(handle):
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read().decode('utf-8'))
    contents = contents.rstrip()
    return len(contents.split('\n\n'))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile", default=None, help="filename in the conll format")
    args = argparser.parse_args()
    if args.inputfile is None:
        logging.warning("using standard input")
        print(countSentences(sys.stdin))
    elif args.inputfile[-3:] == '.gz':
        with gzip.open(args.inputfile) as f:
            print(countSentences(f))
    else:
        with open(args.inputfile) as f:
            print(countSentences(f))
