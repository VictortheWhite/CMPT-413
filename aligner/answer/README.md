
# Assignment 3: Aligner
	* this program is written in Python3
	* Please run it like "python3 answer/align.py -p europarl -f de -n 100000 > output.a"
------------------

## IBM Model 2

### Algorithm Overview
#### initialization
* $t\,(f_i, e_{a_i}) = \frac{c(f_i, e_{a_i})}{ c(f_i) \,\times \,c(e_{a_i}) }$
* $a(i, j, I, L) = \frac {1}{L}$

	
#### training
* for each iteration
	* for k = 1...n
		* for i = 1...I
			* for j = 1...J
				* $\delta(k, i, j) = \frac{t(f_{i}^{(k)}|\,f_{j}^{(k)})\,a(j|i, I, J)}{\sum_{j=0}^{J}\,t(f_{i}^{(k)}|\,f_{j}^{(k)})\,a(j|i, I, J)}$
				* $c(e_{j}^{(k}, f_{i}^{(k)}) += \delta(k, i, j)$
				* $c(e_{j}^{(k}) += \delta(k, i, j)$
				* $c(j|i, I, J) += \delta(k, i, j)$
				* $c(i, I, J) += \delta(k, i, j)$
	* recalculate parameters
		* $t(f|e) = \frac{c(e, f)}{c(e)}$
		* $q(j|i, I, J) = \frac{c(j|i, I, J)}{c(i, I, J}$

#### Align
* for each f
	* $j = argmax \,\,t(f|e) \, a(i | j, I, J)$
	* output (i, j)
* Align using Pr(f|e) and Pr(e|f) by taking intersection


#### Smoothing
	* add-n smoothing is used in this assignment




