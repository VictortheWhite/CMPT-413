
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

    def __getSubSequentEntries(self, scentence ,entry):
        """get all entries for each word matches scentence at position entry.start"""
        #print "new entry: ", entry.word
        entries = []
        maxLenOfSegment = 10
        i = entry.endIndex
        wordLen = 0
        word = ""
        while wordLen <= maxLenOfSegment and i < len(scentence):
            #print i, wordLen, len(scentence)
            word = "".join(scentence[entry.endIndex:i])
            #print word, self.probDict(word)
            if self.probDict(word):
                newEntry = self.Entry(word, i, entry.logProb+math.log(self.probDict(word)), entry)
                entries.append(newEntry)
            wordLen += 1
            i += 1
        word = "".join(scentence[entry.endIndex:i])
        #print word, self.probDict(word)
        if self.probDict(word):
            newEntry = self.Entry(word, i, entry.logProb+math.log(self.probDict(word)), entry)
            entries.append(newEntry)
        return entries

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


    """
    def segment(self, scentence):
        chart = [None]*(len(scentence)+1)            # dp table for argmax of each prefix
        heap = []                             # a queue containing entries to be expanded

        startEntry = self.Entry(None, 0, 0, None)
        heapq.heappush(heap, startEntry)

        while heap:
            e = heapq.heappop(heap)
            print e.endIndex
            #print e.endIndex
            if e.endIndex >= 0:
                #print e.endIndex, chart[e.endIndex]
                if chart[e.endIndex]:
                    if e.logProb > chart[e.endIndex].logProb:
                        chart[e.endIndex] = e
                else:
                    chart[e.endIndex] = e

            for newEntry in self.__getSubSequentEntries(scentence, e):
                heapq.heappush(heap, newEntry)

        # get the best segmentation
        res = []
        entry = chart[-1];
        while entry:
            if entry.word:
                res.append(entry.word)
            entry = entry.lastEntry
        res.reverse()
        return res
    """


if __name__ == '__main__':
    # the default segmenter does not use any probabilities, but you could ...
    Pw1  = Pdist(opts.counts1w)
    Pw2 = Pdist(opts.counts2w)

    ugram = Unigram(Pw1)

    """
    with open(opts.input) as f:
        for line in f:
            scentence = line.strip().split()
            output = ugram.segment(scentence)
            print " ".join(output)
    """

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
