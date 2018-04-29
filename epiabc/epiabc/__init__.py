import numpy as np
import scipy
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
    samp = np.zeros(run, dtype = float)
    weight = np.zeros(run, dtype = float)
    simcount, i = 0, 0
    # sampold and weightold are the lambda values and importance weight from the
    # previous run.
    V=np.sum(sampold**2*weightold)/np.sum(weightold)-np.sum(sampold*weightold)**2/np.sum(weightold)**2
    ss=np.sqrt(2*V)
    while i<run:
        simcount+=1
        ll = np.random.choice(sampold, 1, p = weightold/weightold.sum(), replace = True)
        Lambda = np.random.normal(ll, ss, 1)
        if(Lambda >= 0):
            m = SIR_sim(N, Lambda, 1)
            if abs(m-mstar)<=epsil:
                samp[i] = Lambda
                # Computes the importance weight for the parameter based upon an Exp(1) prior.
                weight[i] = np.exp(-Lambda)/np.sum(weightold*norm.pdf(Lambda,sampold,ss))
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
            for j in range(np.floor(Z)):
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
        for i in range(1, n):
            y.append(t[i-1]/np.sum(q[0:i]))
        q = max(y[0:(m-1)])
        if q<y[m-1]:           
            output[j,:] = np.array([q, y[m-1]])
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

def PCOUP_SMC(Xdata, epss, k, run, OX):
    """ Code for running SMC-ABC for partially coupled household epidemics"""
    #initialize empty zero array
    OUTPUT = np.zeros(((2*epss[1]+1)*run, 6), dtype = float)
    simcount=0
    count=0
    jj=0
    #Setting standard deviation for importance sampling
    VL = np.sum(OX[:,2]**2*OX[:,3])-np.sum(OX[:,2]*OX[:,3])**2
    sL = np.sqrt(2*VL)
    # Number of samples stored from run acceptances.
    cox = OX[:, 0].size
    
    while jj<run:
        simcount+=1
        LA = np.random.choice(cox, 1, p = OX[:,3]/OX[:,3].sum(), replace = True)[0]
        lambda_L = np.random.normal(OX[LA-1, 2], sL, 1)[0]
        if lambda_L>0:
            J=House_COUP(Xdata,epss[1],lambda_L,k) # Run coupled simulations
            W = J[J[:,3]<J[:, 4],:]# W contains successful simulations (infect close to xA individuals)
            if W.size == 5:
                W = np.array(W).reshape((-1, 5))
            if W[:,0].size>0:
                if  min(W[:, 1])<=epss[0]:
                    jj+=1
                    for ii in range(W[:,0].size):
                        if W[ii, 1]<=epss[0]:
                            count+=1
                            OUTPUT[count-1,:] = np.r_[W[ii, 1:5], lambda_L, np.exp(-lambda_L)/sum(norm.pdf(lambda_L, OX[:,2], sL))]
            
        
    return {'OUTPUT':OUTPUT[0:count,:], 'simcount':simcount}


def PCOUP(Xdata, epss, k, run):
    """Partially coupled ABC algorithm to obtain run accepted values."""
     #initialize empty zero array
    output = np.zeros(((2*epss[1]+1)*run, 5), dtype = float)
    simcount=0
    count=0
    jj=0
    
    while jj<run:
        simcount+=1
        lambda_L = np.random.exponential(1,1) # Sample lambda_L
        J = House_COUP(Xdata,epss[1],lambda_L,k) # Run coupled simulations
        W = J[J[:,3]<J[:,4],:]        # W contains successful simulations (infect close to xA individuals)
        if W.size == 5:
            W = np.array(W).reshape((-1, 5))
        if W[:,0].size > 0:
            if min(W[:, 1])<=epss[0]:
                jj+=1
                for ii in range(W[:, 0].size):
                    if W[ii, 1]<=epss[0]:
                        count+=1
                        output[count-1,:]=np.r_[W[ii,1:5], lambda_L]
                        print(jj,count,simcount)
                        
    # Stores values from simulation - these include closeness of simulated epidemic 
    # to data, range of lambda_G values and lambda_L
    return {'OUTPUT':output[0:count,:], 'simcount':simcount}



def Mexp(k,Lambda,U,a,b):
    """Moment calculator for Exp(\lambda) prior."""
    if k == 0:
        U+=np.exp(-a*Lambda)-np.exp(-Lambda*b)
    if k>0:
        U+=a**k*np.exp(-Lambda*a)-b**k*np.exp(-Lambda*b)+k/Lambda*Mexp((k-1), Lambda, U, a, b)
        
    return U

def House_imp(Xdata,epsil,k,run,OX):
    """Code for running ABC for households with importance sampling"""
    OUTPUT = np.zeros((run, 3))
    hs = np.sum(Xdata, axis = 0)
    isB = np.sum(Xdata, axis = 1)
    rowB = np.arange(hs.shape[0]+1)
    xB = np.sum(np.multiply(isB, rowB))
    
    simcount, j = 0, 0
    #Compute variances for the pertubations. Bivariate Gaussian
    meanG = np.sum(np.multiply(OX[:,0], OX[:,2]))/np.sum(OX[:,2])    
    meanL = np.sum(np.multiply(OX[:,1], OX[:,2]))/np.sum(OX[:,2])
    vG = np.sum(np.multiply(OX[:,0]**2, OX[:,2]))/np.sum(OX[:,2]) - meanG**2
    vL = np.sum(np.multiply(OX[:,0]**2, OX[:,2]))/np.sum(OX[:,2]) - meanL**2
    vLG = np.sum(OX[:,0]*OX[:,1]*OX[:,2])/np.sum(OX[:,2]) - meanG*meanL
    vaR = 2*np.array([vG, vLG, vLG, vL]).reshape((2,2)).T
    Sinv = np.linalg.inv(vaR)
    sG = np.sqrt(2*vG)
    sLL = np.sqrt(vL-vLG**2/vG)
    sAL = vLG/vG
    
    while j<run:
        simcount+=1
        LA = np.random.choice(run, 1, p = OX[:,2]/OX[:,2].sum(), replace = True)[0]
        lambda_G = np.random.norm(OX[LA-1, 0], sG, 1)[0]
        lambda_L = np.random.norm((OX[LA-1, 1]+sAL*(lambda_G-OX[LA-1, 0])), sLL, 1)[0]
        
        if lambda_G>0 and lambda_L>0:
            J = House_SEL(hs,lambda_G,lambda_L,k)
            if np.sum(abs(J-Xdata))<=epsil[1]:
                j+=1
                weiZ = 0
                for i in range(run):
                    xDIF = np.array([lambda_G, lambda_L])-OX[LA-1, 0:2]
                    mult = xDIF.T @ Sinv @ xDIF
                    weiZ+=np.exp(-mult[0, 0]/2)
            weiG = np.exp(-1*(lambda_L+lambda_G))/weiZ
            OUTPUT[j-1,:] = [lambda_G,lambda_L,weiG]
    
    return {'OUTPUT':OUTPUT, 'simcount':simcount}

                
def House_COUP(Xdata,epsil,lambda_L,k):
    """Partially coupled ABC algorithm for household epidemics
    lambda_L is drawn from the prior (or however)
    Code finds lambda_G values consistent with the data
    Input: Xdata - Epidemic data to compare simulations with.
    epsil - Max distance between simulated and observed final size for a simulation 
    to be accepted. (Tighter control on distance after simulations straightforward).
    lambda_L - local infection (household) rate
    k - Gamma(k,k) infectious period with k=0 a constant infectious period."""
    hsA = np.sum(Xdata, axis = 0) # hsA[i] - Number of households of size i
    isA = np.sum(Xdata, axis = 1) # isA[i] - Number of households with i-1 infectives
    colA = np.arange(1, hsA.shape[0]+1)
    rowA = np.arange(0, hsA.shape[0]+1)
    
    xA = np.sum(isA*rowA) # Final size
    HH = hsA.shape[0] # HH maximum household size
    ks = np.arange(1, HH+1)
    
    n = np.repeat(ks, hsA)
    m = n.shape[0]# Number of households
    N = np.sum(n) # Population size
    NS = N.copy() # Number of susceptibles
    sev=0       # Running tally of severity (sum of infectious periods)
    threshold=0 # Running tally of (global) threshold required for the next infection
    
    ni = np.repeat(0, n.size) # infectives (per household)
    ns = n.copy() # susceptibles (per household)
    
    OUT = np.zeros((HH+1, HH))
    OUT[0, :] = hsA # Epidemic data in the same form as Xdata
                   # Start with everybody susceptible
    
    DISS = np.zeros((2*epsil+1, 5)) # Matrix for collecting epidemics infecting within epsil of xA infectives.
    SEVI = np.zeros((N, 3)) # Matrix to keep track of number of infectives, severity and threshold.
    ys=0    # number of infectives
    count=0 # number of global infections taking place. First global infection is the introductory case. 
    
    while ys<=(xA+epsil):
        # Only need to consider the epidemic until xA+epsil infections occur.
        # We simulate successive global infections (should they occur) with associated
        # local (within household epidemics)
        # For the count+1 global infection to take place, we require that 
        # for k=1,2,..., count;  lambda_G * severity from first k infectives is larger
        # than the k^th threshold
        count+=1
        kk = np.random.choice(m, 1, p = ns/ns.sum(), replace = True)[0]
        OUT[ni[kk-1],n[kk-1]-1]=OUT[ni[kk-1],n[kk-1]-1]-1
        hou_epi=House_epi(ns[kk-1],k,lambda_L)# Simulate a household epidemic among the remaining susceptibles in the household
        
        ns[kk-1]-=hou_epi[0]
        ni[kk-1] = n[kk-1] - ns[kk-1]#update household kk data (susceptibles and infectives)
        
        OUT[ni[kk-1],n[kk-1]-1]+=1# Update the state of the population following the global 
        #infection and resulting household epidemic
        NS = ns.sum()
        threshold+=np.random.exponential(size = 1, scale = (N/NS))
        
        ys+=hou_epi[0]
        sev+=hou_epi[1]
        if count<=N:
            SEVI[count-1,:] = [ys,sev,threshold]
        # If the number infected is close to xA, we check what value of lambda_G 
        # would be needed for an epidemic of the desired size. 
        # Note that in many cases no value of lambda_G will result in an epidemic 
        # close to xA. 
        if abs(ys-xA)<=epsil:
            dist = np.sum(abs(OUT-Xdata))
            TT = SEVI[0:count, 2]/SEVI[0:count, 1] #ratio of threshold to severity
            Tlow = max(TT[0:(count-1)])
            Thi=TT[0:count].max()   #  Thi is the maximum lambda_G which leads to at most count global infections
            DISS[ys-(xA-epsil), :] = [1,dist,abs(ys-xA),Tlow,Thi]
            
    return DISS

            
##################################
# Coupled-ABC Homogeneously mixing SIR code
###################################
def epidemic(n, m, run):
    """Simulates epidemics with a constant infectious period length 1.
    output gives the set of lambda parameters consistent with the data (m out of n infected)
    Only successful simulations kept """
    output = np.zeros((run, 2))
    ss = np.arange(1, n)
    count = 0
    for j in range(run):
        t = thres(n)
        y = t/ss
        q = y[0:m-1].max()
        if q<y[m-1]:
            count+=1
            output[count-1,:] = [q, y[m-1]]
   
    return output[0:count, :]

def epigamma(n, m, run, k):
    """Simulates epidemics with a Gamma(k,k) infectious period.
    output gives the set of lambda parameters consistent with the data (m out of n infected)
    Only successful simulations kept"""
    output = np.zeros((run, 2), dtype = float)
    count = 0
    for j in range(run):
        t = thres(n)
        q = np.random.gamma(k, 1/k, n)
        y = []
        for i in range(1, n):
            y.append(t[i-1]/np.sum(q[0:i]))
            
        q = max(y[0:(m-1)])
        if q<y[m-1]:
            count+=1
            output[count-1, :] = np.array([q, y[m-1]])
            
    return output[0:count,:]


            
def process(DATA):
    hsA = np.sum(DATA,axis=0)
    isA = np.sum(DATA,axis=1)
    colA = np.array(range(1,len(hsA)+1))
    rowA = np.array(range(0,len(hsA)+1))
    
    xA = np.sum(isA*rowA) # Final size
    N = np.sum(hsA*colA) # Population size
    
    return hsA, isA, colA, rowA, xA, N

def House_SEL(hs, lambda_G, lambda_L, k):
    HH = hs.size
    ks = np.array(range(1,HH+1))

    n = np.repeat(ks,hs)

    m = n.size  # Number of households
    N = n.sum() # Population size
    NS = N

    sev = 0
    threshold = 0

    ni = np.repeat(0,m)
    ns = n.copy()

    R = np.random.exponential(size=N)

    while threshold <= lambda_G*sev:
        kk = np.random.choice(a=np.arange(m)+1, size = 1, replace = True, p = ns/(ns.sum()))[0]
        
        hou_epi = House_epi(ns[kk-1],k,lambda_L)
        ns[kk-1] = ns[kk-1] - hou_epi[0]
        sev = sev + hou_epi[1]
        
        ni[kk-1] = n[kk-1]-ns[kk-1] 
        #NS=NS-hou_epi[0]
        NS=ns.sum()
        
        if NS > 0:
            threshold = threshold + np.random.exponential(scale = N/NS, size=1)[0]
        if NS == 0:
            threshold = 2*lambda_G*sev
        
    OUT = np.zeros((HH+1,HH))
    for i in range(1, HH+2):
        for j in range(1,HH+1):
            OUT[i-1,j-1] = sum(n[ni==(i-1)]==j)
    
    return OUT
#
# Code for running vanilla ABC for households
#
def House_van(Xdata,epsil,k,run):
    """
    Xdata - a data set
    epsil - a numpy array of length 2
    k - int
    run - int
    """
    
    OUTPUT = np.zeros((run,2))
    hs = np.sum(Xdata,axis=0).astype('int')
    isB = np.sum(Xdata,axis=1)
    
    rowB = np.arange(hs.size+1)
    xB = sum(isB*rowB)
    
    simcount = 0
    j = 0
    
    while j<run:
        simcount = simcount+1
        lambda_G = np.random.exponential(size=1)
        lambda_L = np.random.exponential(size=1)
        J = House_SEL(hs,lambda_G,lambda_L,k)
        
        if ((abs(J-Xdata)).sum()) <= epsil[0]:
            isJ = np.sum(J, axis=1)
            if abs(sum(isJ*rowB)-xB) <= epsil[1]:
                j = j+1
                OUTPUT[j-1,] = np.array([lambda_G,lambda_L]).reshape((1,2))
                print(j,simcount)
                
    return {'OUTPUT': OUTPUT, 'simcount' : simcount}

def House_epi(n,k,lambda_L):
    
    i=0
    sev=0
    
    if n == 1:
        i = 1
        if k == 0:
            sev = 1
        if k > 0:
            sev = np.random.gamma(k, 1/k, 1)
    
    if n > 1:
        t = thresH(n)
        if k == 0:
            q = np.repeat(1.0,n)
        if k > 0:
            q = np.random.gamma(size=n, shape=k, scale=1/k)
        t = np.append(t, 2*lambda_L*np.sum(q))
        
        i = 0
        test = 0
        while test == 0:
            i = i + 1
            if t[i-1] > (lambda_L*np.sum(q[0:i])):
                test = 1 
                sev = np.sum(q[0:i])
                
    return np.array([i,sev])
# Code for setting (local) thresholds in a household of size n.
# 
# Note local thresholds not required for households of size 1.
# Infection rate does not depend upon households size.
#

def thresH(n):
    thres = np.repeat(0.0,n-1)
    thres[0] = np.random.exponential(size=1,scale=1/(n-1))
    if n > 2:
        for i in range(2,n):
            thres[i-1] = thres[i-2] + np.random.exponential(size=1, scale=1/(n-i))
    
    return thres

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


def simSIR_constrained(N, beta, gamma, final_size):
    check = 0
    while check == 0:
        out = simSIR(N,beta,gamma)
        if out['final.size'] >= final_size:
            res = out['removal.times']
            check = 1
    
    return res


def abcSIR(obs_data, N, epsilon, prior_param, samples):
    
    # first retrieve the final size of the observed data
    final_size_obs = obs_data.size
    
    # matrix to store the posterior samples
    post_samples = np.nan * np.zeros((samples,2))
    
    i = 0
    while i < samples:
        
        # draw from the prior distribution
        beta = np.random.exponential(size = 1, scale = 1/prior_param[0])   
        gamma = np.random.exponential(size = 1, scale = 1/prior_param[1]) 
        
        # simulate data
        sim_data = simSIR(N, beta, gamma)
      
        #check if the final size matches the observedata
        if sim_data['final.size'] == final_size_obs:
            d = sum((obs_data - sim_data['removal.times'])**2)
            
            if d < epsilon:
                i = i+1
                post_samples[i-1,] = np.array((beta,gamma))
    
    return post_samples



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




def simSIR_discrete(N, Lambda, gamma, T):
    
    # change the rate to avoidance probability
    q = np.exp(-Lambda/N)
    
    # initialisation
    t_vec = np.arange(1,T+1)
    
    It_vec = np.repeat(np.nan, T)   
    St_vec = np.repeat(np.nan, T) 
    Rt_vec = np.repeat(np.nan, T) 
    
    It_vec[0] = 1
    St_vec[0] = N - 1
    Rt_vec[0] = 0
    
    # sample infectious period for the initially infective individual
    inf_per = np.random.geometric(p=gamma, size=1) + 1; 
   
    # Yt.vec keeps track of the number of people being removed on each day
    Yt_vec = np.repeat(0, T)
   
    # assing the value to Yt which corresponds to when the initially infective individual gets removed
    Yt_vec[inf_per + 1] = Yt_vec[inf_per + 1] + 1  
    
    t = 1
    while t < T:
        
        t = t + 1
        # simulate the number of new infections for next day t
        
        if It_vec[t-2] > 0:
            
            new_inf = np.random.binomial(size = 1, n = St_vec[t-1], p = 1-q**It.vec[t-1])
            St_vec[t-1] = St_vec[t-2] - new_inf
            It_vec[t-1] = It_vec[t-2] + new_inf - Yt_vec[t-1] 
            
            if new_inf > 0:
                for j in range(1,new_inf+1):
                    inf_per = np.random.geometric(p=gamma, size=1) + 1
                    loc = min(t+0+inf_per, T)
                    Yt_vec[loc-1] = Yt_vec[loc-1] +1
                   
        else:
            It_vec[t-1] = It_vec[t-2]
            St_vec[t-1] = St_vec[t-2]
    
    Rt_vec = np.cumsum(Yt_vec)
    res = np.r_[It_vec, St_vec, Rt_vec, Yt_vec]

    if (sum(np.sum(res[0:res.shape[1],],axis=0) - np.repeat(N, repeats= T))!=0):
    	return('error') 
   
    return {"pop" : res, "final.size" : sum(res[3,])} 




def abcSIR_discrete(obs_data, N, T, epsilon, prior_param, samples):
    
    # first retrieve the final size of the observed data
    final_size_obs = obs_data.size
    
    # matrix to store the posterior samples
    post_samples = np.nan * np.zeros((samples,2))
    
    i = 0
    
    while i < samples:
        
        Lambda = np.random.exponential(scale=1/prior_param[0], size=1)
        gamma = np.random.uniform(size = 1)
        
        # simulate data
        sim_data = simSIR_discrete(N, Lambda, gamma, T)
        
        # get start/end dates
        start_date = min(np.where(sim_data['pop'][3,] == 1))  
        end_date = start_date + len_out - 1
        
        d = sum((obs_data - sim_data['pop'][3, ][np.arange(start_date-1,end_date)])**2)
        
        if d < epsilon:
            i = i+1
            post_samples[i-1,] = np.array((Lambda, gamma))
            
        return post_samples