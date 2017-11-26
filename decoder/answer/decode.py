#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple


def decode(f):
    # The following code implements a monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of
    # the first i words of the input sentence. You should generalize
    # this so that they can represent translations of *any* i words.
    hypothesis = namedtuple("hypothesis", "logprob, bitstr, lm_state, predecessor, phrase")
    initial_hypothesis = hypothesis(0.0, '0'*len(f), lm.begin(), None, None)
    stacks = [{} for _ in f] + [{}]
    stacks[0][initial_hypothesis.lm_state] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        sys.stderr.write("\tdecoding with len %d/%d \n" % (i, len(f)))
        for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
            for s, t, p, bitstr in phrase_generator(f, h.bitstr):
                if p in tm:
                    j = i + len(p)
                    for phrase in tm[p]:
                        logprob = h.logprob + phrase.logprob
                        lm_state = h.lm_state
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            logprob += word_logprob
                        logprob += lm.end(lm_state) if j == len(f) else 0.0
                        new_hypothesis = hypothesis(logprob, bitstr, lm_state, h, phrase)
                        if (lm_state, bitstr, t) not in stacks[j] or stacks[j][(lm_state, bitstr, t)].logprob < logprob: # second case is recombination
                            stacks[j][lm_state, bitstr, t] = new_hypothesis
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    # keep it though i don't know what it does lol
    if opts.verbose:
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

    return extract_english(winner)


# helper methods
'''
def phrase_generator(s, bitstr):
    def generatePhrases(s, bitstr, i):
        if i == len(s):
            return [((), '')]
        phrases = generatePhrases(s, bitstr, i+1)
        if bitstr[i] == '1':
            return [(p, '1'+b_str) for (p, b_str) in phrases]
        else:
            return [((s[i],) + p, '1'+b_str) for p, b_str in phrases] + [(p, '0'+b_str) for (p, b_str) in phrases]
    return generatePhrases(s, bitstr, 0)
'''
def phrase_generator(s, bitstr):
    for i in range(len(s)):
        if bitstr[i] == '1':
            continue
        for j in range(i, len(s)):
            if bitstr[j] == '1':
                break
            yield i, j, s[i:j+1], bitstr[:i]+'1'*(j-i+1)+bitstr[j+1:]



def extract_english(h):
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=10, type="int", help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=500, type="int", help="Maximum stack size (default=1)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
    opts = optparser.parse_args()[0]

    tm = models.TM(opts.tm, opts.k)
    lm = models.LM(opts.lm)
    french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

    # tm should translate unknown words as-is with probability 1
    for word in set(sum(french,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]

    sys.stderr.write("Decoding %s...\n" % (opts.input,))
    for i, f in enumerate(french):
        sys.stderr.write("Decoding sentence %d \n" % i)
        print decode(f)

    #print f
    #print len(f)
    #print len(phrase_generator(f, '0'*len(f)))
    #for i, j, p, b in phrase_generator(('how', 'are', 'you'), '100'):
    #    print i, j, p, b

#    print phrase_generator(range(20), '0'*20);
