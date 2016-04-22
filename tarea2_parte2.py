import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

u=np.random.uniform(0,1,100000)
x=norm.ppf(u)

plt.hist(x,100)
plt.show()

