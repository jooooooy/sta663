import numpy as np
import math
from scipy.stats import norm, gamma, poisson

def bernSIR(n, beta, gamma, p):
    t, count = 0, 0
    # Matrix of edges - non-symmetric but ok because we will only use MAT[i,j] if
    # i infected before j or MAT[j,i] if j infected before i.
    MAT = np.random.binomial(1, p, n**2).reshape((n, n))
    rowM = np.sum(MAT, axis = 1)
    # Set individual 1 infectious and everybody else susceptible.
    I = np.zeros(n)
    I[0] = 1
    output = np.zeros(n) # Recovery times
    while np.sum(I == 1) > 0:
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
        else:
            S = np.zeros(n)
            S[I==1] = 1
            K = np.random.choice(n, 1, p = S/S.sum(), replace = True)
            I[K] = 2
            count+=1
            output[count-1]=t
            
    return output, count    

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
