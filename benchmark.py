
########################################################################################
# Code for implementing the Toni et al. alogrithm with predefined levels of
# approximation.
#
# Implementation for the Abakiliki data
########################################################################################
tstartC = time.time()
N = 120
mstar = 30
epsil = [10,5,2,1,0]
TT = len(epsil)

# Estimate of posterior mean and variance at each threshold.
EstMean = np.repeat(0.0, TT)
EstSD = np.repeat(0.0, TT)
# Total number of simulations
SIMtotal = np.repeat(0, TT)
run = 100
output=ABCrej(N,mstar,epsil[0],run)
samp=output['samp']
simTotal=output['simcount']
EstMean[0] = np.mean(samp)
EstSD[0] = np.std(samp)
SIMtotal[0] = simTotal
# Initial weights all equal
weight = np.repeat(1.0, run)

for t in range(1, TT):
    output = ABCimp(N,mstar,epsil[t],run,samp,weight)
    samp = output['samp']
    weight = output['weight']
    simTotal+=output['simcount']
    
    SIMtotal[t] = simTotal
    EstMean[t]=sum(samp*weight)/sum(weight)
    EstSD[t]=math.sqrt(sum(samp**2*weight)/sum(weight)-EstMean[t]**2)
    
print(EstMean)
print(EstSD)
print(SIMtotal)

tendC = time.time()
print('running time is:', tendC-tstartC)
    

