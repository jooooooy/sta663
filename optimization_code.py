import numpy as np
import math
from scipy.stats import norm, gamma, poisson
import operator
import pandas as pd
import sys
from functools import reduce
from itertools import product
import numba
from numba import jit

@jit
def thres_numba(n):
    thres = np.zeros(n-1)
    thres[0] = np.random.exponential(n/(n-1), 1)[0]
    for i in range(1, n-1):
        thres[i] = thres[i-1] + np.random.exponential(n/(n-i-1), 1)[0]
        
    return thres
    
 

