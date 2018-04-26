
# coding: utf-8

# In[ ]:


import numpy as np
import math
from scipy.stats import norm, gamma, poisson


# In[44]:


SeattleA = np.array([[15,12,4],[11,17,4],[0,21,4],[0,0,5]])
SeattleA


# In[63]:


Tecumseh1 = np.zeros((8,7))
Tecumseh1[0,0:5] = np.array((66 ,87 ,  25  , 22   , 4 ))
Tecumseh1[1,0:5] = np.array((13 ,14 ,  15  ,  9   , 4 ))
Tecumseh1[2,1:6] = np.array((4 ,   4  ,  9  ,  2 ,   1 ))
Tecumseh1[3,2:7] = np.array(( 4   , 3   , 1  ,  1 ,   1))
Tecumseh1[4,3:5] = np.array(( 1,1))
Tecumseh1


# In[45]:


#
# Process data
#

def process(DATA):
    hsA = np.sum(DATA,axis=0)
    isA = np.sum(DATA,axis=1)
    colA = np.array(range(1,len(hsA)+1))
    rowA = np.array(range(0,len(hsA)+1))
    
    xA = np.sum(isA*rowA) # Final size
    N = np.sum(hsA*colA) # Population size
    
    return {'hsA' : hsA, 'isA' : isA, 'colA' : colA, 'rowA' : rowA, 'xA' : xA, 'N' : N}

process(SeattleA)


# In[4]:


# Code for setting (local) thresholds in a household of size n.
# 
# Note local thresholds not required for households of size 1.
# Infection rate does not depend upon households size.
#

def thresH(n):
    thres = np.repeat(0.0,n-1)
    thres[0] = np.random.exponential(size=1,scale=1/(n-1))
    if n > 2:
        for i in range(1,n-1):
            thres[i] = thres[i-1] + np.random.exponential(size=1, scale=1/(n-i))
    
    return thres

thresH(5)


# In[5]:


def House_epi(n,k,lambda_L):
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
            i += 1
            if t[i-1] > (lambda_L*np.sum(q[0:i])):
                test = 1 
                sev = np.sum(q[0:i])
                
    return np.array([i,sev])


# In[6]:


House_epi(1,3,0.21)


# In[7]:


House_epi(3,5,0.21)


# In[8]:


#
# Code for simulating a household epidemic using the Sellke construction
#

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
            threshold = threshold + np.random.exponential(scale = NS/N, size=1)[0]
        if NS == 0:
            threshold = 2*lambda_G*sev
        
    OUT = np.zeros((HH+1,HH))
    for i in range(1, HH+2):
        for j in range(1,HH+1):
            OUT[i-1,j-1] = sum(n[ni==(i-1)]==j)
    
    return OUT


# In[9]:


hs = np.array([1,2,3])
lambda_G = 0.9
lambda_L = 0.21
k=5
House_SEL(hs, lambda_G, lambda_L, 0)


# In[38]:


OUTPUT = np.zeros((run,2))
hs = np.sum(Xdata,axis=0).astype('int')
isB = np.sum(Xdata,axis=1)
rowB = np.arange(hs.size+1)
xB = sum(isB*rowB)
simcount = 0
j = 0


# In[39]:


simcount = simcount+1
simcount


# In[40]:


lambda_G = np.random.exponential(size=1)
lambda_L = np.random.exponential(size=1)
lambda_G, lambda_L


# In[47]:


J = House_SEL(hs,lambda_G,lambda_L,k)
J


# In[48]:


(abs(J-Xdata)).sum()


# In[50]:


isJ = np.sum(J, axis=1)
isJ


# In[60]:


j = j+1
OUTPUT[j-1,] = np.array([lambda_G,lambda_L]).reshape((1,2))


# In[61]:


OUTPUT[j-1,]


# In[86]:


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
            


# In[87]:


Xdata = Tecumseh1
epsil = np.array((100,10))
k = 0
run = 10
House_van(Xdata,epsil,k,run)


# In[83]:


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
    


# In[96]:


N=10
beta=0.9
gamma=0.21
out = simSIR(N, beta, gamma)
out


# In[97]:


# this is function that simulates an outbreak of a predetermined final size -- to be used for the
# Abakaliki data

def simSIR_constrained(N, beta, gamma, final_size):
    check = 0
    while check == 0:
        out = simSIR(N,beta,gamma)
        if out['final.size'] >= final_size:
            res = out['removal.times']
            check = 1
    
    return res

simSIR_constrained(N, beta, gamma, final_size=5)


# In[101]:


# first retrieve the final size of the observed data
final_size_obs = obs_data.size

# matrix to store the posterior samples
post_samples = np.zeros((samples,2))

i = 0


# In[102]:


beta = np.random.exponential(size = 1, scale = 1/prior_param[0])   
gamma = np.random.exponential(size = 1, scale = 1/prior_param[1]) 
beta,gamma


# In[151]:


sim_data = simSIR(N, beta, gamma)
sim_data


# In[168]:


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


# In[ ]:


obs_data = np.array((2,6,3,7, 8, 4, 0))
N = 120
epsilon = 11
prior_param = np.array((0.1,0.1))
samples = 500


# In[ ]:


abcSIR(obs_data, N, epsilon, prior_param, samples)


# In[163]:


sum((obs_data - sim_data['removal.times'])**2)


# In[ ]:


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
        sim_data_binned = np.array(sum(sim.data['removal.times']<=breaks_data[1]))
        for i in range(1,len(breaks_data)-1):
            sim_data_binned = np.append(sim_data_binned, sum((breaks_data[i]<times) & (times<=breaks_data[i+1])))
        
        #check if the final size matches the observedata
        d = np.sqrt( sum((obs_data_binned - sim_data_binned)**2) + ((obs_duration - sim_duration)/50)**2 )
        
        if d < epsilon:
            i = i + 1
            print(i)
            post_samples[i-1,] = np.array((beta,gamma))
         
    print(K)
    post_samples


# In[ ]:


obs_data = np.array((2,6,3,7, 8, 4, 0))
N = 120
epsilon = 11
prior_param = np.array((0.1,0.1))
samples = 500
obs_duration = 76
breaks_data = np.array((0, 13, 26, 39, 52, 65, 78, np.inf))
breaks_data


# In[ ]:


abcSIR_binned(obs_data, breaks_data, obs_duration, N, epsilon, prior_param, samples)


# In[ ]:


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
        break 
   
    return {"pop" : res, "final.size" : sum(res[3,])} 


# In[ ]:


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
        start_date = min(np.where(sim_data['pop'][3,] == 1)) +1   min(which(sim_data['pop'][3,] == 1)) 
        end_date = start_date + len_out - 1
        
        d = sum((obs_data - sim_data['pop'][3, ][np.arange(start_date-1,end_date)])**2)
        
        if d < epsilon:
            i = i+1
            post_samples[i-1,] = np.array((Lambda, gamma))
            
        return post_samples

