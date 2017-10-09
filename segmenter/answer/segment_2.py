
import sys, codecs, optparse, os
import heapq
import math

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
optparser.add_option("-u", "--unigram", dest="useUnigram", action="store_true", default=False, help="use unigram instead of bigram")
(opts, _) = optparser.parse_args()

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return float(self[key])/float(self.N)
        #else: return self.missingfn(key, self.N)
        elif len(key) == 1: return self.missingfn(key, self.N)
        else: return None

class Unigram(object):
    """Unigram Segementer"""
    def __init__(self, probDict):
        self.probDict = probDict

    class Entry:
        def __init__(self, word, endIndex, logProb, lastEntry):
            self.word = word
            self.endIndex = endIndex
            self.logProb = logProb
            self.lastEntry = lastEntry

        def __lt__(self, other):
            return self.endIndex < other.endIndex

    def segment(self, scentence):
        chart = [None] * (len(scentence)+1)
        startEntry = self.Entry(None, 0, 0, None)
        chart[0] = startEntry

        for i in range(1, len(chart)):
            maxProb = -float('inf')
            for j in reversed(range(0, i)):
                word = "".join(scentence[j:i])
                if self.probDict(word):
                    prob = chart[j].logProb + math.log(self.probDict(word))
                    if prob > maxProb:
                        chart[i] = self.Entry(word, i, prob, chart[j])

        # get the best segmentation
        e = chart[-1]
        output = []
        while e:
            if e.word:
                output.append(e.word)
            e = e.lastEntry
        output.reverse()
        return output


class Bigram(object):
    """Bigram Segementer"""
    def __init__(self, probDict_U, probDict_B):
        self.probDict_U = probDict_U
        self.probDict_B = probDict_B

    def segment(self, scentence):



        output = []
        return output



if __name__ == '__main__':
    # the default segmenter does not use any probabilities, but you could ...
    Pw1  = Pdist(opts.counts1w)
    Pw2 = Pdist(opts.counts2w)

    ugram = Unigram(Pw1)
    
    old = sys.stdout
    sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
    # ignoring the dictionary provided in opts.counts
    with open(opts.input) as f:
        for line in f:
            utf8line = unicode(line.strip(), 'utf-8')
            input_words = [i for i in utf8line]
            output = ugram.segment(input_words)  # segmentation is one word per character in the input
            print " ".join(output)
    sys.stdout = old
