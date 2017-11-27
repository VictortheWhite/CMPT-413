## Decoder

#### Algorithm Overview
* let hypothesis be a tuple of (logprob, bitstr, lm_state, predecessor, phrase)
* for stack i from 0 to len(f)
    * for hypothesis in pruned(stack[i])
        * for new_bitstr, new_phrase in phrase_generator(hypothesis.bitstr)
            * calculate new log probability and generate new hypothesis
            * store of update hypothesis in stack[j]
                * key is a tuple of (lm_state, bitstr, t) where t is the ending word in f
                * hypothesis will be recombined in this step
* decode and output result from the hypothesis with highest score in last stack


#### phrase_generator
* The phrase_generator will find all possible phrases composed with 1 or more consecutive untranslated word and new bitstr, given the old bitstr which tells which word hasn't been translated.

#### What we've tried
* Add distortion
    * distortion is defined as how far the startpos of phrase is from endpos of last phrase. Long distance will be panelized.
    * This properties doesn't work well in our practice. It is not kept in out final solution.
* estimate future cost
    * compute a future cost table, using translation model and unigram model. This can be easily done using dynamic programming.
    * The future cost part is not well debugged and has negative impact. It's not adopted in our example. But I believe it should be really useful if properly implemented

#### test results
* Our best score is -1228.206, which is the highest on leaderboard for now. It takes some time less than 1 hour to compute all sentences. The parameter used is '-s 5000'
* An acceptable parameter is '-s 200'. It gives a score of -1270.629, which is pretty good. And all sentences will be decoded within 1 or 2 minutes.
