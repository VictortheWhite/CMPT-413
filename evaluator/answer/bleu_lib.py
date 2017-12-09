#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import math
import sys

from collections import Counter
from nltk.util import ngrams

def sentence_bleu(candidate, reference, weights):
    r_len = len(reference)
    c_len = len(candidate)

    # Collects the various precision values for the different ngram orders.
    p_n = [(modified_precision(candidate, reference, i + 1)) for i, _ in enumerate(weights)]

    # Smoothen the modified precision.
    p_n = smoothing(p_n)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(r_len, c_len)

    return bp * geometric_mean(p_n)

def geometric_mean(precisions):
    m = 1.0
    for p in precisions:
        m *= p
    return m ** (1.0 / len(precisions))

def modified_precision(candidate, reference, n):
    # Extracts all ngrams in hypothesis
    candidate_ngrams_counts = Counter(ngrams(candidate, n)) if len(candidate) >= n else Counter()
    reference_ngrams_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()

    # Calculate max counts
    max_counts = {}
    for ngram in candidate_ngrams_counts:
        max_counts[ngram] = max(max_counts.get(ngram, 0), reference_ngrams_counts[ngram])

    # Calculate clipped counts
    clipped_counts = {ngram: min(count, max_counts[ngram]) for ngram, count in candidate_ngrams_counts.items()}

    # The precision will be numerator / denominator, here we return those two value
    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(candidate_ngrams_counts.values())) # avoid ZeroDivisionError.
    return numerator, denominator

def brevity_penalty(r, c):
    return 1 if c > r else math.exp(1 - (r / c))

def smoothing(p, epsilon=0.1):
    """
    Smoothing method: Add *epsilon* counts to precision with 0 counts.
    """
    # numerator: # of ngram matches.
    # denominator: # of ngram in ref.
    return [(numerator + epsilon) / denominator if numerator == 0 else numerator / denominator for numerator, denominator in p]
