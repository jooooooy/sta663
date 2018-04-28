
# coding: utf-8

# In[2]:


import numpy as np
import math
from scipy.stats import norm, gamma, poisson
import operator
import pandas as pd
import sys
from functools import reduce
import time


# In[1]:


def bernSIRC(n, beta, gamma, p, CUT):
    t, count = 0, 0
    MAT = np.random.binomial(1, p, n**2).reshape((n, n))
    rowM = np.sum(MAT, axis = 1)
    # Set individual 1 infectious and everybody else susceptible.
    I = np.zeros(n)
    I[0] = 1
    output = np.zeros(n) # Recovery times
    cut_count = 1
    while np.sum(I == 1) > 0 and cut_count < CUT:
        rec = np.sum(I == 1)
        infe = np.sum(rowM[I == 1])
        t+= np.random.exponential(1/gamma*rec+beta*infe, 1)
        u = np.random.uniform(0, 1, 1)
        if u <= beta*infe/(gamma*rec+beta*infe):
            S = np.zeros(n)
            S[I==1] = rowM[I==1]
            K = np.random.choice(np.arange(n), 1, p = S/S.sum(), replace = True)
            J = np.random.choice(np.arange(n), 1, p = MAT[K,].ravel()/MAT[K,].ravel().sum(), replace = True)
            if I[J] == 0:
                I[J] = 1
                cut_count+=1
        else:
            S = np.zeros(n)
            S[I==1] = 1
            K = np.random.choice(n, 1, p = S/S.sum(), replace = True)
            I[K] = 2
            count+=1
            output[count-1]=t
            
    if cut_count==CUT:
        count=cut_count
        
    return output, count


# In[3]:


sin = time.time()
N = 10
OUT = np.zeros((N,89))
PARA = np.zeros((N,5))

for i in range(1,N):
    
    p = np.random.uniform(size=1)
    beta = np.random.exponential(scale = 1/2, size=1)
    gamma = np.random.gamma(shape = 2, size = 1, scale = 1)
    
    Xout = bernSIRC(89, beta, gamma, p,40)
    OUT[i-1,] = Xout[0]
    PARA[i-1,] =  np.r_[p, beta, gamma, Xout[1], OUT[i-1,Xout[1]-1]]

sout = time.time()
sout - sin

