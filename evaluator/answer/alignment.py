class Alignment(object):
    def __init__(self, wordStrings1, wordStrings2):
        self.line1 = wordStrings1
        self.line2 = wordStrings2
        self.matches = [[] for _ in range(len(self.line2))]
        self.line1Coverage = [0 for _ in range(len(self.line1))]
        self.line2Coverage = [0 for _ in range(len(self.line2))]
