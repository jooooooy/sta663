import numpy as np
import math
from scipy.stats import norm, gamma, poisson

def ABCimp(N, mstar, epsil, run, sampold, weightold):
    """ABC importance sampling for a fixed number of accepted values.
    Use this in implementing the Toni et al. algorithm."""
    samp = np.zeros(run)
    weight = np.zeros(run)
    simcount, i = 0, 0
    # sampold and weightold are the lambda values and importance weight from the
    # previous run.
    V=np.sum(sampold^2*weightold)/np.sum(weightold)-np.sum(sampold*weightold)^2/np.sum(weightold)^2
    ss=math.sqrt(2*V)
    while i<run:
        simcount+=1
        ll = np.random.choice(sampold, 1, p = weightold/weightold.sum(), replace = True)
        Lambda = np.random.normal(ll, ss, 1)
        if(Lambda >= 0):
            m = SIR_sim(N, Lambda, 1)
            if abs(m-mstar)<=epsil:
                samp[i] = Lambda
                # Computes the importance weight for the parameter based upon an Exp(1) prior.
                weight[i] = np.exp(-1*Lambda)/np.sum(weightold*norm.pdf(Lambda,sampold,ss))
                i+=1
     
    return {'samp': samp, 'weight': weight, 'simcount': simcount}


def SIR_sim(N, Lambda, k):
    """Function for simulating SIR epidemic"""
    # Gamma distributed infectious period  - Gamma (k,k)
    # (k=0 - constant infectious period)
    S = N-1
    y = 1
    while y>0:
        y-=1
        if k == 0:
            I = 1
        if k>0:
            I = np.random.gamma(k, 1/k, 1)[0]
        #draw possion distribution    
        Z = np.random.poisson(Lambda*I, 1)
        if Z>0:
            for j in range(math.floor(Z)):
                u = np.random.uniform(0, 1, 1)
                if u<(S/N):
                    S-=1
                    y+=1
    return N-S 

def ABCrej(N, mstar, epsil, run):
    """ABC rejection sampler for a fixed number of accepted values."""
    samp = np.zeros(run)
    simcount, i = 0, 0
    while i<run:
        simcount+=1
        Lambda = np.random.exponential(1, 1)
        m = SIR_sim(N, Lambda, 1)
        if abs(m-mstar)<=epsil:
            samp[i] = Lambda
            i+=1
     
    return {'samp':samp, 'simcount':simcount}

def epigammaF(n, m, run, k):
    """Simulates epidemics with a Gamma(k,k) infectious period.""" 
    # output gives the set of lambda parameters consistent with the data (m out of n infected)
    # Only successful simulations kept 
    # k=0 constant infectious period
    output = np.zeros((run, 2))
    count, j = 0, 0
    while j<run:
        t = thres(n)
        count+=1
        if k>0:
            q = np.random.gamma(k, 1/k, n)
        if k == 0:
            q = np.repeat(1, n)
        y = []
        for i in range(n-1):
            y.append(t[i]/np.sum(q[0:(i-1)]))
        q = np.nanmax(y[0:(m-1)])
        if q<y[m-1]:           
            output[j,] = np.array([q, y[m-1]])
            j+=1
        
    return output

def thres(n):
    thres = np.zeros(n-1)
    thres[0] = np.random.exponential(n/(n-1), 1)[0]
    for i in range(1, n-1):
        thres[i] = thres[i-1] + np.random.exponential(n/(n-i-1), 1)[0]
        
    return thres

def dpow(MM, k):
    return (np.sum(MM[:,1]**k)-np.sum(MM[:,0]**k))/k

def meanC(MM):
    return dpow(MM,2)/dpow(MM,1)
    
def varC(MM):
    return dpow(MM,3)/dpow(MM,1)-meanC(MM)**2

def sdC(MM):
    return varC(MM)**(1/2)
