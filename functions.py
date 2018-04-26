import numpy as np
import math
from scipy.stats import norm, gamma, poisson
import operator
import pandas as pd
import sys
from functools import reduce
from itertools import product

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
    output = np.zeros(((2*epss[1]+1)*run, 6))
    simcount=0
    count=0
    jj=0
    #Setting standard deviation for importance sampling
    VL=np.sum(OX[:,2]^2*OX[:,3])-np.sum(OX[:,2]*OX[:,3])^2
    sL=sqrt(2*VL)
    # Number of samples stored from run acceptances.
    cox = OX[:, 0].shape[0]
    
    while jj<run:
        simcount+=1
        LA = np.random.choice(cox, 1, p = OX[:,3]/OX[:,3].sum(), replace = True)[0]
        lambda_L = np.random.normal(OX[LA-1, 2], sL, 1)[0]
        if lambda_L>0:
            J=House_COUP(Xdata,epss[1],lambda_L,k) # Run coupled simulations
            W = J[J[:,3]<J[:, 4],:]# W contains successful simulations (infect close to xA individuals)
            if W.shape[1] == 5:
                W = np.array(W).reshape(1, 5)
            if W[:,0].shape[0]>0:
                if np.amin(W, axis = 0)[1] <=epss[0]:
                    jj+=1
                    for ii in range(W[:,0].shape[0]):
                        if W[ii, 1]<=epss[0]:
                            count+=1
                            OUTPUT[count-1,:] = reduce(operator.concat, [W[ii, 1:4], lambda_L, np.exp(-1*lambda_L)/norm.pdf(lambda_L, OX[:,2], sL)])
            
        
    return {'OUTPUT':OUTPUT[0:count,:], 'simcount':simcount}


def PCOUP(Xdata, epss, k, run):
    """Partially coupled ABC algorithm to obtain run accepted values."""
     #initialize empty zero array
    output = np.zeros(((2*epss[1]+1)*run, 5))
    simcount=0
    count=0
    jj=0
    
    while jj<run:
        simcount+=1
        lambda_L = np.random.exponential(1,1) # Sample lambda_L
        J = House_COUP(Xdata,epss[1],lambda_L,k) # Run coupled simulations
        W = J[J[:,3]<J[:,4],:]        # W contains successful simulations (infect close to xA individuals)
        if W.shape[1] == 5:
            W = np.array(W).reshape(1, 5)
        if W[:,0].shape[0]>0:
                if np.amin(W, axis = 0)[1] <=epss[0]:
                    jj+=1
                    OUTPUT[count-1,:] = reduce(operator.concat, [W[ii, 1:4], lambda_L]) 
                
    # Stores values from simulation - these include closeness of simulated epidemic 
    # to data, range of lambda_G values and lambda_L
    return {'OUTPUT':OUTPUT[0:count,:], 'simcount':simcount}

def Mexp(k,Lambda,U,a,b):
    """Moment calculator for Exp(\lambda) prior."""
    if k == 0:
        U+=math.exp(-a*Lambda)-math.exp(-Lambda*b)
    if k>0:
        U+=a**k*math.exp(-Lambda*a)-b**k*math.exp(-Lambda*b)+k/Lambda*Mexp((k-1), Lambda, U, a, b)
        
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
    sG = math.sqrt(2*vG)
    sLL = math.sqrt(vL-vLG**2/vG)
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
            weiG = math.exp(-1*(lambda_L+lambda_G))/weiZ
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
    
    n = np.repeat(ks, hsA, axis=0)
    m = n.shape[0]# Number of households
    N = np.sum(n) # Population size
    NS = N # Number of susceptibles
    sev=0       # Running tally of severity (sum of infectious periods)
    threshold=0 # Running tally of (global) threshold required for the next infection
    
    ni = np.repeat(0, n.shape[0], axis = 0) # infectives (per household)
    ns=n # susceptibles (per household)
    
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
        ni[kk-1]-=ns[kk-1]#update household kk data (susceptibles and infectives)
        
        OUT[ni[kk-1],n[kk-1]-1]+=1# Update the state of the population following the global 
        #infection and resulting household epidemic
        NS = ns.sum()
        threshold+=np.random.exponential(1, (N/NS))
        
        ys+=hou_epi[0]
        sev+=hou_epi[1]
        SEVI[count-1,:] = [ys,sev,threshold]
        # If the number infected is close to xA, we check what value of lambda_G 
        # would be needed for an epidemic of the desired size. 
        # Note that in many cases no value of lambda_G will result in an epidemic 
        # close to xA. 
        if abs(ys-xA)<=epsil:
            dist = np.sum(abs(OUT-Xdata))
            TT = SEVI[0:count, 2]/SEVI[0:count, 1] #ratio of threshold to severity
            Tlow = TT[0:(count-1)].max()
            Thi=TT[0:count].max()   #  Thi is the maximum lambda_G which leads to at most count global infections
            DISS[(ys-(xA-epsil)), :] = [1,dist,abs(ys-xA),Tlow,Thi]
            
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

def binning1d(x, y, breaks, nbins):
    """binning functions"""
    
    f = pd.cut(x, breaks, retbins = True)
    if any(f.isnull()):
        sys.exit('breaks do not span the range of x')
        
    freq = pd.cut(x, breaks, labels = False)
    midpoints = (breaks[1:] + np.delete(breaks, nbins))/2
    x = midpoints[freq > 0]
    x.freq = freq[freq > 0]
    if not all(np.isnan(y)):
        means = pd.DataFrame({'x': x, 'y': y})['y'].groupby([f]).mean()
        sums = pd.DataFrame({'x': x, 'y': y})['y'].groupby([f]).sum()
        devs = pd.DataFrame({'x': x, 'y': y})['y'].groupby([f]).var()
    
    return {'x': x, 'x.freq': x.freq, 'table.freq' = freq, 'breaks': breaks, 'means': means, 'sums': sums, 'devs': devs}
        
def binning2d(x, y, breaks, nbins):
    f1 = pd.cut(x[:, 0], breaks[:, 0], retbins = True)
    f2 = pd.cut(x[:, 1], breaks[:, 1], retbins = True)
    freq1 =  pd.cut(x[:, 0], breaks[:, 0], labels = False)
    freq2 = pd.cut(x[:, 1], breaks[:, 1], labels = False)
    freq = pd.crosstab(freq2, freq1)
    midpoints = (breaks[1:, :] + np.delete(breaks, nbins, axis = 0))/2
    z1 = midpoints[:, 0]
    z2 = midpoints[:, 1]
    X = reduce(operator.concat, [np.repeat(z1, z2.shape[0]), np.repeat(z2, np.repeat(z1.shape[0], z2.shape[0]))]) 
    X.f = freq.T.values.ravel()
    ID = X.f > 0
    X = X[ID, :]
    X.f = X.f[ID]
    if not all(np.isnan(y)):
        means = pd.DataFrame({'x': x, 'y': y})['y'].groupby([f1, f2]).mean()
        devs = pd.DataFrame({'x': x, 'y': y})['y'].groupby([f1, f2]).var()
    
    return {'x': X, 'x.freq': X.f, 'table.freq' = freq, 'breaks': breaks, 'means': means, 'devs': devs}
        
def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())], 
                        columns=dictionary.keys())

def binning3d(x, y, breaks, nbins):
    f1 = pd.cut(x[:, 0], breaks[:, 0], retbins = True)
    f2 = pd.cut(x[:, 1], breaks[:, 1], retbins = True)
    f3 = pd.cut(x[:, 2], breaks[:, 2], retbins = True)
    freq1 = pd.cut(x[:, 0], breaks[:, 0], labels = False)
    freq2 = pd.cut(x[:, 1], breaks[:, 1], labels = False)
    freq3 = pd.cut(x[:, 2], breaks[:, 2], labels = False)
    freq = pd.crosstab(f1, [f2, f3])
    midpoints = (breaks[1:, :] + np.delete(breaks, nbins, axis = 0))/2
    z1 = midpoints[:, 0]
    z2 = midpoints[:, 1]
    z3 = midpoints[:, 2]
    X = expand_grid({'z1': z1, 'z2':z2, 'z3':z3})
    X.f = freq.values.ravel()
    ID = (X.f > 0)
    X = X[ID, :]
    if not all(np.isnan(y)):
        means = pd.DataFrame({'x': x, 'y': y})['y'].groupby([f1, f2, f3]).mean()
        devs = pd.DataFrame({'x': x, 'y': y})['y'].groupby([f1, f2, f3]).var()
        
    return {'x': X, 'x.freq': X.f, 'table.freq' = freq, 'breaks': breaks, 'means': means, 'devs': devs}

def binning(x, y, breaks, nbins):
    """Binning functions for 1-3D"""
    if len(x.shape) > 0:
        if len(x.shape)!=2:
            sys.exit("wrong parameter x for binning")
        ndim = x.shape[1]
        if ndim > 3:
            sys.exit("binning can be carried out only with 1-3 variables")
        if y is None:
            y = np.repeat(np.nan, x.shape[0])
        if nbins is None:
            nbins = round(math.log(x.shape[0])/math.log(2)+1)
        if breaks is None:
            breaks = np.hstack([np.arange(min(x[:, 0], max(x[:, 0]+1, nbins+1))).reshape((-1,1)), 
                                    np.arange(min(x[:, 1], max(x[:, 1]+1, nbins+1))).reshape((-1,1))])
            if ndim == 3:
                breaks = np.hstack([breaks, np.arange(min(x[:, 2], max(x[:, 2]+1, nbins+1))).reshape((-1,1))])
                breaks[0,:] = breaks[0,:] - np.repeat(10**-5, breaks.shape[1]) 
        else:
            nbins = breaks.shape[0]-1
        if np.isnan(max(abs(breaks))) or max(abs(breaks)) is None:
            sys.exit("illegal breaks")
            
        if ndim == 2:
            result = binning2d(x, y, breaks = breaks, nbins = nbins)
        else:
            result = binning3d(x, y, breaks = breaks, nbins = nbins)
            
    else:
        x = x.ravel()
        if y is None:
            y = np.repeat(np.nan, len(x))
        if nbins is None:
            nbins = round(math.log(x.shape[0])/math.log(2)+1)
            breaks[0] = breaks[0] - 10**-5
        else:
            nbins = len(breaks) - 1
        if np.isnan(max(abs(breaks))) or max(abs(breaks)) is None:
            sys.exit("illegal breaks")
            
        result = binning1d(x, y, breaks = breaks, nbins = nbins)
    
    return result
            
            
