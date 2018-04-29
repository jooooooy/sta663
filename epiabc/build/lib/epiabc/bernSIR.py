import numpy as np
import time

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