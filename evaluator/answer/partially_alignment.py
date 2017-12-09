class PartialAlignment(object):
    def __init__(self, matches, line1UsedWords, line2UsedWords):
        self.matches = matches
        self.line1UsedWords = line1UsedWords
        self.line2UsedWords = line2UsedWords
        self.matchCount = 0
        self.matches1 = 0
        self.matches2 = 0
        self.chunks = 0
        self.idx = 0
        self.lastMatchEnd = -1
        self.distance = 0

    def isUsed(self, match):
        if self.line2UsedWords[match.start]:
            return True
        if self.line1UsedWords[match.matchStart]:
            return True
        return False

    def setUsed(self, match, b):
        self.line2UsedWords[match.start] = b
        self.line1UsedWords[match.matchStart] = b

    def __repr__(self):
        return '<PartialAlignment: ' + str(self.matches) + ' / ' + str(self.matchCount) + ' / ' + str(self.matches1) + ' / ' + str(self.matches2) + ' / ' + str(self.chunks) + ' / ' + str(self.lastMatchEnd) + ' / ' + str(self.idx) + ' / ' + str(self.distance) + '>'
