class Match(object):
    def __init__(self, start, length, matchStart, matchLength, prob):
        self.start = start # start of the match(line2)
        self.matchStart = matchStart # start of this match(line1)
        self.prob = prob # probability supplied by matcher

    def __repr__(self):
        return str(self.start) + '->' + str(self.matchStart)
