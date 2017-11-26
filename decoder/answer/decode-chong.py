#!/usr/bin/env python
import argparse
import sys
import models
from collections import namedtuple


def calculate_ph(state, f, tm):
  ph_list = []
  for s in range(0, len(f)):
    for t in range(s+1, len(f)+1):
      if (abs(state.r + 1 - s) > args.distortion_limit):
        break
      if f[s:t] in tm:
        bitstring = 0
        for i in range(s, t):
          bitstring += pow(2, len(f)-1-i)
        if ((bitstring & state.bitString) == 0):
          ph_list.append(ph(s, t, tm[f[s:t]]))
  return ph_list


def next(prev_state, ph_i, lm_state, f):
  bitstring = prev_state.bitString
  e1 = None
  e2 = None
  if len(lm_state) == 1:
    e2 = (lm_state[0],)
  elif len(lm_state) == 2:
    e1 = (lm_state[0],)
    e2 = (lm_state[1],)
  for i in range(ph_i.s, ph_i.t):
    bitstring = bitstring | pow(2, len(f)-1-i)
  return state(e1, e2, bitstring, ph_i.t)


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument("-i", "--input", dest="input", default="data/input",
                         help="File containing sentences to translate (default=data/input)")
  argparser.add_argument("-t", "--translation-model", dest="tm", default="data/tm",
                         help="File containing translation model (default=data/tm)")
  argparser.add_argument("-l", "--language-model", dest="lm", default="data/lm",
                         help="File containing ARPA-format language model (default=data/lm)")
  argparser.add_argument("-o", "--output", dest="output", default="output",
                         help="Ouput result file")
  argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=2**64, type=int,
                         help="Number of sentences to decode (default=2^64)")
  argparser.add_argument("-k", "--translations-per-phrase", dest="k", default=2**64, type=int,
                         help="Limit on number of translations to consider per phrase (default=2^64)")
  argparser.add_argument("-b", "--beam-size", dest="beam_size", default=1000, type=int,
                         help="Maximum beam size (default=1000)")
  argparser.add_argument("-dl", "--distortion-limit", dest="distortion_limit", default=10, type=int,
                         help="Hard distortion limit (default=10)")
  argparser.add_argument("-dp", "--distortion-parameter", dest="distortion_parameter", default=-0.01, type=float,
                         help="Soft distortion parameter (default=-0.01)")
  args = argparser.parse_args()

  tm = models.TM(args.tm, args.k)
  lm = models.LM(args.lm)
  french = [tuple(line.strip().split()) for line in open(args.input, 'r', encoding='utf-8').readlines()[:args.num_sents]]
  output = open(args.output, 'w', encoding='utf-8')

  # tm should translate unknown words as-is with probability 1
  for word in set(sum(french,())):
    if (word,) not in tm:
      tm[(word,)] = [models.phrase(word, 0.0)]

  hypothesis = namedtuple("hypothesis", "logprob, state, predecessor, phrase")
  state = namedtuple("state", "e1, e2, bitString, r")
  ph = namedtuple("ph", "s, t, phrase")

  for f in french:
    initial_state = state(None, lm.begin(), 0, 0.0)
    initial_hypothesis = hypothesis(0.0, initial_state, None, None)
    Q = [{} for _ in range(len(f) + 1)]
    Q[0][initial_state] = initial_hypothesis

    for i, Q_i in enumerate(Q[:-1]):
      for q in sorted(Q_i.values(), key=lambda q: -q.logprob)[:args.beam_size]: # beam size prune
        prev_state = q.state
        for ph_i in calculate_ph(prev_state, f, tm):
          for phrase in ph_i.phrase:
            # calculate language model logprob
            lm_logprob = 0
            lm_state = ()
            lm_state += prev_state.e1 if prev_state.e1 is not None else ()
            lm_state += prev_state.e2 if prev_state.e2 is not None else ()
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              lm_logprob += word_logprob
            next_state = next(prev_state, ph_i, lm_state, f)
            length = bin(next_state.bitString).count("1")
            lm_logprob += lm.end(lm_state) if length == len(f) else 0.0
            # calculate logprob
            logprob = q.logprob + phrase.logprob + lm_logprob + args.distortion_parameter * abs(prev_state.r + 1 - ph_i.s)
            # update Q if needed
            new_hypothesis = hypothesis(logprob, next_state, q, phrase)
            if next_state not in Q[length] or Q[length][next_state].logprob < logprob: # second case is recombination
              Q[length][next_state] = new_hypothesis

    winner = max(Q[-1].values(), key=lambda q: q.logprob)
    def extract_english(q):
      return "" if q.predecessor is None else "%s%s " % (extract_english(q.predecessor), q.phrase.english)
    english = extract_english(winner)
    print(english)
    output.write(english + '\n')
