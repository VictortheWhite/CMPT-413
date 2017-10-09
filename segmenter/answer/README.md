# Word segmenter

* In this assignment, 2 models are implemented: unigram and bigram
* This program is written in Python3, please use Python3 to run it
* Unigram is used by default, to use Unigram, run "Python3 answer/segment.py --unigram"

## Unigram
 * Unigram possibility of a segmentation is defined as:
    * <img src="http://chart.googleapis.com/chart?cht=tx&chl=   P_{w}(w_0^i ) \times P_{w} (w_{i+1}^j ) \times \;...\; \times P_{w} (w_n^{n - k} )" style="border:none;">
    <br>
 * The algorithm finds a segmentation that maximizes it, that is
    * <img src="http://chart.googleapis.com/chart?cht=tx&chl= argmax \; P_{w}(w_0^i ) \times P_{w} (w_{i+1}^j ) \times \;...\; \times P_{w} (w_n^{n - k} )" style="border:none;">
    <br />
 * Algorithm Overview
    * A recursive relation can be used to solve it by dynamic programming:
        * <img src="http://chart.googleapis.com/chart?cht=tx&chl= segment( c_0, \, ... \, ,c_i ) = argmax P_w (w_j^i) \times segment(c_0, \, ... \, ,c_{j-1}) \;where \;segment(\emptyset) \, = \, 1.0" style="border:none;">
    * let chart[i] stores the sequence that ends at i that maximizes the score function. Initialize chart[0] with 1
    * for i from 1 to len(word)
        * for every word that ends at i
            * for every sequence that prefixes word
                * find the word and it's prefix that maximizes chart[i], that is, max(log(Pw(word)) + prob(prefix)), where prob(prefix) is stored in chart[j] with j being the end index of the prefix
    * return the final entry of chart, which is the sequence that maximizes the score function

## Bigram

 * Bigram possibility of a segmentation is defined as:
    * <img src="http://chart.googleapis.com/chart?cht=tx&chl= P_w(w_1 ,\, ... \, , w_n ) = p_w(w_1) \times p_w(w_2 | w_1) \times, \, ... \, ,\times P_w(w_n\,|\,w_{n-1})" style="border:none;">
    <br />
 * Algorithm Overview
    * A recursive relation can be used to solve it by dynamic programming:
        * <img src="http://chart.googleapis.com/chart?cht=tx&chl= segment(w_{i}) = segment(w_{i-1}) \times P_w(w_i | w_{i-1})" style="border:none;">
    * let chart[i] store the sequence of word that maximizes the score function, ending with i
    * for i from 1 to len(word):
        * for every word that ends at i
            * for every sequence that prefixes word (input_words[startPos:i])
                * find the word and it's prefix (chart[startPos-1]) that maximizes chart[i], using the conditional chain of possibility, store the sequence and prob in chart[i]
    * return the final entry of chart, which store the wanted sequence

## Smoothing function
    * the smoothing function is defined as followed:

    ```py
    def _smoothing_func(self, key):
        """Better smoothing function"""
        if len(key) <= 1:
            return 1. / self.totalvalue
        else:
            score = 1. / self.totalvalue
            for i in range(1, len(key)):
                score = score / (args.smooth * i * self.totalvalue)
            return score + 1e-200 # avoid log(0)
    ```

    * The idea of this smoothing function is that, when an unknown word appears, the longer it is, the possibility of it should fall quickly
    * an arg is used to control the speed that it falls.
