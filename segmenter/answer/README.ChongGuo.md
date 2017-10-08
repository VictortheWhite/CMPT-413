# Homework 1 - Word Segmentation

## Chong Guo - 301295753

------------------

* I first try to implement the baseline **iteratively** version word segmentation method according to the pseudo-code on the course website **(unigram with laplace smoothing)**, but unfortunately I only get an accuracy of 84.63%. [(Github link)](https://github.com/Armour/CMPT413-Assignments/commit/a8624cefee51f74700e2024b787b653ee1fd5562)

* Then I added `generator.py` which used to read the `wseg_simplified_cn.txt` and generate extra input file for unigram and bigram called `count_1w_extra.txt` and `count_2w_extra.txt`, but unfortunately extra input data made the accuracy even worse :( [(Github link)](https://github.com/Armour/CMPT413-Assignments/commit/74b37dc9d86359bc3354e0092b985690570dc016)

* Because we expect baseline method should be about 88.637%, so I decide to debug my code and output the final chart result for input, then I realize there is actually a problem with my current implementation:

    ```bash
    ==> Final chart entry:  (0, '法', 0, -8.161607601006384)
    ==> Final chart entry:  (1, '正', 1, -7.846526554366489)
    ==> Final chart entry:  (3, '研究', 2, -7.252751847619747)
    ==> Final chart entry:  (4, '从', 4, -6.669952724228667)
    ==> Final chart entry:  (5, '波', 5, -10.358832178342603)
    ==> Final chart entry:  (6, '波黑', 5, -8.321950251081564)
    ==> Final chart entry:  (8, '撤军', 7, -9.154859374016667)
    ==> Final chart entry:  (9, '计', 9, -10.358832178342603)
    ==> Final chart entry:  (10, '计划', 9, -7.346570602837401)
    ```

    notice that **chart[2] is missing**.

* To fix this problem, I changed to use the dynamic programming way to implement the unigram method, and finally reach accuracy 88.55% [(Github Link)](https://github.com/Armour/CMPT413-Assignments/commit/84f951677f10f444693b7071246a6559dd536e12)

    the idea is that for each i, to compute chart[i].prob, choose the best j (j < i) with max chart[i-j].prob + log(prob(word[i-j+1:i+1])):

    ```py
    # chart[i] stores a pair of (word, prob)
    for i in range(len(self.input_words)):
        for j in range(1, args.maxlen + 1):
            if i - j + 1 < 0:
                continue
            word = "".join(self.input_words[i-j+1:i+1])
            prob = math.log(self.prob_dist(word))
            prev_prob = self.chart[i - j][1] if i - j >=0 else 0
            if i not in self.chart or (prev_prob + prob) > self.chart[i][1]:
                self.chart[i] = (word, prev_prob + prob)
    ```

* After finished the unigram version, I also implemented the bigram version dynamic programming method. [(Github Link)](https://github.com/Armour/CMPT413-Assignments/commit/56a4a6700b39a1af26169b713d9faa90f4498787)

    ```py
    def get_probability(self, word1, word2):
        word_pair = word1 + ' ' + word2
        if word_pair in self.prob_dist2 and word1 in self.prob_dist:
            # Laplacian bigram probabilities
            return math.log((self.prob_dist2.count(word_pair) + 1) / (self.prob_dist.count(word1) + self.prob_dist.totaltype))
        else:
            # Backoff
            return math.log(self.prob_dist(word2))

    # chart[(i, j)] means end with index i and the last word has length j, it stores a pair of (word, prob)
    for i in range(args.maxlen):
        word1 = ""
        word2 = "".join(self.input_words[:i+1])
        prob = self.get_probability(word1, word2)
        self.chart[(i, len(word2))] = (word2, prob)

    for i in range(1, len(self.input_words)):
        for j in range(1, args.maxlen + 1): # the length of the current word
            if i - j + 1 < 0:
                continue
            for k in range(1, args.maxlen + 1): # the length of the prev word
                if i - j - k + 1 < 0:
                    continue
                word1 = "".join(self.input_words[i-j-k+1:i-j+1])
                word2 = "".join(self.input_words[i-j+1:i+1])
                prob = self.get_probability(word1, word2)
                prev_prob = self.chart[(i - j, k)][1]
                if (i, j) not in self.chart or (prev_prob + prob) > self.chart[(i, j)][1]:
                    self.chart[(i, j)] = (word2, prev_prob + prob)
    ```

* The accuracy of my bigram version is almost the same as the unigram version, so I decided its time to change the smoothing function to get a better score, and here comes my magic smoothing function for unknown word with magic parameter! [(Github Link)](https://github.com/Armour/CMPT413-Assignments/commit/0a7024efca4bdbd86d72b04096bfa2fb946aa301)

    ```py
    def _smoothing_func(self, key):
        """Better smoothing function"""
        if len(key) <= 1:
            return 1. / self.totalvalue
        else:
            score = 1. / self.totalvalue
            # I think the longer the unknown word is, the less chance it will be, and the chance will decreasing more and more faster when i increasing, the only thing I need to do is to adjust the smooth parameter
            for i in range(1, len(key)):
                score = score / (args.smooth * i * self.totalvalue)
            return score + 1e-200 # avoid log(0)
    ```

    Through adjust the smooth parameter, I can get a stable result within 92% ~ 94%, the best case I tried is when **smooth = 0.0245**, the result is **94.218%** using unigram, and **94.114%** using bigram, and finally we got the first place on leaderboard lol.

* Tbh, I don't think this result is generalized for other testing data, it's kind of keep adjusting parameter for a better result. I think if we have more time, we could find a more general way to solve this kind of problem with a satisfied result, like RNN maybe.
