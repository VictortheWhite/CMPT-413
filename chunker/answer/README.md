# Phrasal Chunking
* This program is written in python3. perc.py has been modified as well.
* Please run by
    * python3 chunker.py
    * python3 perc.py > output
    * python score-chunks.py -t output


## Averaged Perceptron

#### Motivation
* In perceptron algorithm, the weight is more adapted to the latter examples
* Averaged Perceptron Algorithm, which computes the average of all versions of weight vectors, is better dealing with this problem;

#### Algorithm
* initialization:
    * train data: (x[1:n], t[1:n])
    * weight vector: w (intialized to all zero)
    * summed weight vector: w_sum
* for t in range(numepochs)
    * for i in range(n)
        * <img src="http://chart.googleapis.com/chart?cht=tx&chl= z = argmax\,\Phi \, (x_i, z) \, \cdot w, \, z \in GEN(x)" style="border:none;">
        * if z != t
            * <img src="http://chart.googleapis.com/chart?cht=tx&chl= w = w %2B \Phi (x_i, t_i) - \Phi (x_i, z_i)" style="border:none;">
            * w_sum = w_sum + w
* output averaged weight parameters:
    * w_sum / (t * n)

#### Implementation
* Weight Vector Update:
    * <img src="http://chart.googleapis.com/chart?cht=tx&chl= z = argmax\,\Phi \, (x_i, z) \, \cdot w, \, z \in GEN(x)" style="border:none;">
    * for each feat for x[i]
        * reward w[feat, t[i]] by 1
        * penalize w[feat, z[i]] by 1

* Compute Averaged Vector
    * We didn't calculate a sum and divide by count in the end
    * Instead we compute it on the go
