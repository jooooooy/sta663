import numpy as np

def test():
    print('it works')

    return (1,2,3)



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
        for i in range(1, n):
            y.append(t[i-1]/np.sum(q[0:i]))
        q = max(y[0:(m-1)])
        if q<y[m-1]:           
            output[j,:] = np.array([q, y[m-1]])
            j+=1
        
    return output


def Mexp(k,Lambda,U,a,b):
    """Moment calculator for Exp(\lambda) prior."""
    if k == 0:
        U+=math.exp(-a*Lambda)-math.exp(-Lambda*b)
    if k>0:
        U+=a**k*math.exp(-Lambda*a)-b**k*math.exp(-Lambda*b)+k/Lambda*Mexp((k-1), Lambda, U, a, b)
        
    return U


def thres(n):
    thres = np.zeros(n-1)
    thres[0] = np.random.exponential(n/(n-1), 1)[0]
    for i in range(1, n-1):
        thres[i] = thres[i-1] + np.random.exponential(n/(n-i-1), 1)[0]
    return thres




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



def simSIR(N, beta, gamma):
    # initial number of infectives and susceptibles;
    I = 1
    S = N-1
    
    # recording time;
    t = 0
    times = np.array(0)
    
    # a vector which records the type of event (1=infection, 2=removal)
    type = np.array(1)
    
    while I > 0:
        
        # time to next event;
        t = t + np.random.exponential(size=1,scale= 1/((beta/N)*I*S + gamma*I)) 
        times = np.append(times, t)

        if np.random.uniform(size=1) < beta*S/(beta*S + N*gamma):
            # infection
            I = I+1
            S = S-1
            type = np.append(type,1)
        else:
            #removal
            I = I-1
            type = np.append(type,2)

    return {'removal.times': times[type == 2] - min(times[type == 2]),
           'final.size' : N-S,
           'T' : times[times.size-1] }



def abcSIR_binned(obs_data_binned, breaks_data, obs_duration, N, epsilon, prior_param, samples):
    
    # first retrieve the final size of the observed data
    final_size_obs = obs_data_binned.size
    
    # matrix to store the posterior samples
    post_samples = np.nan * np.zeros((samples,2))

    K = 0
   
    i = 0
    
    while i < samples:
        
        # counter
        K = K + 1
        
        # draw from the prior distribution
        beta = np.random.exponential(size = 1, scale = 1/prior_param[0])   
        gamma = np.random.exponential(size = 1, scale = 1/prior_param[1]) 
        
        # simulate data
        sim_data = simSIR(N, beta, gamma)
        sim_duration = sim_data['T']
        sim_data_binned = np.array(sum(sim_data['removal.times']<=breaks_data[1]))
        for j in range(1,len(breaks_data)-1):
            sim_data_binned = np.append(sim_data_binned, sum((breaks_data[j]<sim_data['removal.times']) & (sim_data['removal.times']<=breaks_data[j+1])))
        
        #check if the final size matches the observedata
        d = np.sqrt( sum((obs_data_binned - sim_data_binned)**2) + ((obs_duration - sim_duration)/50)**2 )
        
        if d < epsilon:
            i = i + 1
            print(i)
            post_samples[i-1,] = np.array((beta,gamma)).reshape((1,2))
         
    print(K)
    return post_samples