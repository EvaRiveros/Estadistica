import math
import numpy as np

def binom(n,k,p):
	comb = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
	return comb*(p**k)*((1.0-p)**(n-k))

prob=binom(33,18,0.5)

a=np.random.binomial(33,0.5,1000)

i=0
for x in xrange(0,999):
	if a[x]==18:
		i+=1


print prob
print i