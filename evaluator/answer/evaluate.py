#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import functools
import copy
import nltk

from itertools import islice, product
from match import Match
from alignment import Alignment
from partially_alignment import PartialAlignment
from nltk.corpus import wordnet

syns_map = {}

def word_matches(h, ref_set):
    return sum(1 for w in h if w in ref_set)

def is_similar_word(w1, w2):
    global syns_map
    word1 = w1.lower()
    word2 = w2.lower()
    if word1 == word2:
        return True
    if word1 in syns_map:
        syns1 = syns_map[word1]
    else:
        syns1 = set(s for s in wordnet.synsets(word1))
        syns_map[word1] = syns1
    if word2 in syns_map:
        syns2 = syns_map[word2]
    else:
        syns2 = set(s for s in wordnet.synsets(word2))
        syns_map[word2] = syns2
    if len(syns1) == 0 or len(syns2) == 0:
        return False
    best = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2)
               for s1, s2 in product(syns1, syns2))
    return best[0] > 0.8

def compare(align1, align2):
    # More matches always wins
    matchDiff = align2.matches1 + align2.matches2 - align1.matches1 - align1.matches2
    if matchDiff > 0:
        return 1
    elif matchDiff < 0:
        return -1
    # Otherwise fewer chunks wins
    chunkDiff = align1.chunks - align2.chunks
    if chunkDiff != 0:
        return chunkDiff
    # Finally shortest distance wins
    return align1.distance - align2.distance

def calculate_chunk(h, ref, beam_size):
    a = Alignment(h, ref)
    line1UsedWords = [False for _ in range(len(a.line1))]
    line2UsedWords = [False for _ in range(len(a.line2))]
    initialPath = PartialAlignment([None for _ in range(len(a.line2))], line1UsedWords, line2UsedWords)

    # init all possible matches
    for i in range(len(a.line1)):
        for j in range(len(a.line2)):
            if is_similar_word(a.line1[i], a.line2[j]):
                a.matches[j].append(Match(start=j, length=1, matchStart=i, matchLength=1, prob=1))
                a.line1Coverage[i] += 1
                a.line2Coverage[j] += 1

    # One-to-one, non-overlapping matches are definite
    for i in range(len(a.matches)):
        if len(a.matches[i]) == 1:
            m = a.matches[i][0]
            overlap = False
            if (a.line2Coverage[i] != 1):
                overlap = True
            if (a.line1Coverage[m.matchStart] != 1):
                overlap = True
            if not overlap:
                initialPath.matches[i] = m
                initialPath.line2UsedWords[i] = True
                initialPath.line1UsedWords[m.matchStart] = True

    # Resolve best alignment using remaining matches
    paths = []
    nextPaths = [initialPath]
    for current in range(len(a.matches) + 1):
        paths = nextPaths
        nextPaths = []
        paths.sort(key=functools.cmp_to_key(compare))
        # print(paths)
        # Try as many paths as beam allows
        numRank = min(beam_size, len(paths))
        for rank in range(numRank):
            path = paths[rank]
            # Case: path is complete
            if current == len(a.matches):
                # Close last chunk
                if path.lastMatchEnd != -1:
                    path.chunks += 1
                nextPaths.append(path)
                continue
            # Case: Current index word is in use
            if path.line2UsedWords[current] is True:
                # If fixed match
                if path.matches[path.idx] != None:
                    m = path.matches[path.idx]
                    path.matchCount += 1
                    path.matches1 += 1
                    path.matches2 += 1
                    # Not continuous in line1
                    if path.lastMatchEnd != -1 and m.matchStart != path.lastMatchEnd:
                        path.chunks += 1
                    # Advance to end of match + 1
                    path.idx = m.start + 1
                    path.lastMatchEnd = m.matchStart + 1
                    path.distance += abs(m.start - m.matchStart)
                    nextPaths.append(path)
                continue
            # Case: Multiple possible matches, for each match starting at index start
            matches = a.matches[current]
            for i in range(len(matches)):
                m = matches[i]
                # Check to see if words are unused
                if path.isUsed(m):
                    continue
                newPath = copy.deepcopy(path)
                # Select m for this start index
                newPath.setUsed(m, True)
                newPath.matches[current] = m
                # Calculate new stats
                newPath.matchCount += 1
                newPath.matches1 += 1
                newPath.matches2 += 1
                # Not continuous in line1
                if newPath.lastMatchEnd != -1 and m.matchStart != newPath.lastMatchEnd:
                    newPath.chunks += 1
                # Advance to end of match + 1
                newPath.idx = m.start + 1
                newPath.lastMatchEnd = m.matchStart + 1
                newPath.distance += abs(m.start - m.matchStart)
                nextPaths.append(newPath)
            # Try skipping this index
            if path.lastMatchEnd != -1:
                path.chunks += 1
                path.lastMatchEnd = -1
            path.idx += 1
            nextPaths.append(path)
        if len(nextPaths) == 0:
            print("Warning: unexpected conditions - skipping matches until possible to continue")
            nextPaths.append(paths[0])
    # Return top best path's chunk number
    nextPaths.sort(key=functools.cmp_to_key(compare))
    return nextPaths[0].chunks, nextPaths[0].matchCount

def meteor(h, ref, beam_size, alpha, beta, gamma):
    ref_set = set(ref)
    chunk_count, match_count = calculate_chunk(h, ref, beam_size)
    if match_count == 0:
        return 0.0
    P = match_count / len(h)
    R = match_count / len(ref)
    F_mean = P * R / (alpha * P + (1 - alpha) * R)
    p = gamma * ((chunk_count / match_count) ** beta)
    return F_mean * (1 - p)

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-s', '--beam_size', default=40, type=int,
            help='Beam search size (default 40)')
    parser.add_argument('-a', '--alpha', default=0.82, type=float,
            help='Tunable parameter alpha (default 0.82)')
    parser.add_argument('-b', '--beta', default=1, type=int,
            help='Tunable parameter beta (default 1)')
    parser.add_argument('-g', '--gamma', default=0.21, type=float,
            help='Tunable parameter gamma (default 0.21)')
    args = parser.parse_args()

    # download wordnet data
    nltk.download('wordnet')

    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(args.input, encoding='utf-8') as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), args.num_sentences):
        h1_score = meteor(h1, ref, args.beam_size, args.alpha, args.beta, args.gamma)
        h2_score = meteor(h2, ref, args.beam_size, args.alpha, args.beta, args.gamma)
        if h1_score > h2_score:
            print(1)
        elif h1_score == h2_score:
            print(0)
        else:
            print(-1)

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
