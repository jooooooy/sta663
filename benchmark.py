import time
import numpy as np
import math
from scipy.stats import norm, gamma, poisson
import operator
import pandas as pd
import sys
from functools import reduce

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
    

#################################################################################################################
# ABC code for running ABC using Toni et al. SMC
#################################################################################################################
SeattleA = np.array([[15,12,4],[11,17,4],[0,21,4],[0,0,5]])
Tecumseh1 = np.zeros((8,7))
Tecumseh1[0,0:5] = np.array((66 ,87 ,  25  , 22   , 4 ))
Tecumseh1[1,0:5] = np.array((13 ,14 ,  15  ,  9   , 4 ))
Tecumseh1[2,1:6] = np.array((4 ,   4  ,  9  ,  2 ,   1 ))
Tecumseh1[3,2:7] = np.array(( 4   , 3   , 1  ,  1 ,   1))
Tecumseh1[4,3:5] = np.array(( 1,1))

tstartC = time.time()

# Set dataset: SeattleA, SeattleB, Tecumseh1, Tecumseh2
DATA = Tecumseh1

# Process data
proc = process(DATA)
hsA, isA, colA, rowA, xA, N = proc
# Code for running coupled algorithm and selecting which simulations to keep.
#
# Set thresholds
Eps1 = [100, 70, 50]
Eps2 = [10, 6, 4]

TT = len(Eps1)
EpsM = np.vstack((Eps1,Eps2)).T

# Set infectious period (k) and  number of iterations (run)
k = 0
run = 1000

# Initial run
epss = EpsM[0,:]

OUTP = House_van(DATA, EpsM[0,:], k, run)
OUT = OUTP['OUTPUT']
simT = OUTP['simcount']

OL = np.zeros((run, 3))
OL[:, 0:2] = OUT
OL[:, 2] = np.repeat(1, run)

# Subsequent runs using importance sampling
if TT > 1:
    for t in range(1, TT):
        OUTP = House_imp(DATA,EpsM[t,],k,run,OL)
        OL = OUTP['OUTPUT']
        simT+=OUTP['simcount']        

OUT = OL


# Moment calculations for SMC-ABC output
weiA = np.sum(OUT[:,2])
meanG = np.sum(OUT[:,0]*OUT[:,2])/weiA
sdG = math.sqrt(np.sum(OUT[:,0]**2*OUT[:,2])/weiA - meanG**2)
meanL = np.sum(OUT[:,1]*OUT[:,2])/weiA
sdL=math.sqrt(np.sum(OUT[:,1]**2*OUT[:,2])/weiA - meanL**2)

# Moment calculations for transformed variables

meanqG = np.sum(math.exp(-OUT[:,0]*xA/N)*OUT[:,2])/weiA
sdqG = math.sqrt(np.sum(math.exp(-2*OUT[:,0]*xA/N)*OUT[:,2])/weiA-meanqG**2)
meanqL = np.sum(math.exp(-OUT[:,1])*OUT[:,2])/weiA
sdqL = math.sqrt(np.sum(math.exp(-2*OUT[:,1])*OUT[:,2])/weiA-meanqL**2)

# Summarise results
print("Summarise results", [meanG, sdG, meanL, sdL])

# Transformed parameters.
print("Transformed parameters.", [meanqG, sdqG, meanqL, sdqL])
print(simT)

tendC = time.time()
print('running time is', tendC-tstartC)

#######################################################################################################################
# Main code for partially coupled ABC for household epidemics
#######################################################################################################################
SeattleA = np.array([[15,12,4],[11,17,4],[0,21,4],[0,0,5]])
tstartC = time.time()
DATA=SeattleA
proc = process(DATA)
hsA, isA , colA, rowA,  xA , N = proc

# Set thresholds
#Eps1=c(100,70,50)
#Eps2=c(10,6,4)

Eps1= [20,12,8]
Eps2= [3,2,1]

TT=len(Eps1)
EpsM= np.vstack((Eps1, Eps2)).T
# Set infectious period (k) and  number of iterations (run)
k=0
run=10

#
# Set infectious period (k) and
# number of iterations (run)
#

OUTP=PCOUP(DATA,EpsM[0,:],k,run)
OUT = OUTP['OUTPUT']
simT = OUTP['simcount']
OL = np.zeros((OUT[:, 0].size,4))
temp = OUT[:,2:5]
OL[:,0:3] = temp
OL[:,3]=np.exp(-OUT[:,2])-np.exp(-OUT[:,3])
OL[:,3]=OL[:,3]/OL[:,3].sum()

if TT > 1:
    for t in range(1, TT):
        OUTP = PCOUP_SMC(DATA,EpsM[t,:],k,run,OL)
        OUT = OUTP['OUTPUT']
        OL = np.zeros((OUT[:, 0].size, 4))
        OL[:, 0:4] = OUT[:, 2:6]
        OL[:, 3] = (np.exp(-OUT[:,2])-np.exp(-OUT[:,3]))*OL[:,3]
        OL[:, 3] = OL[:, 3]/OL[:, 3].sum()
        simT+=OUTP['simcount']
 
# Compute posterior mean and standard deviation
wei = 0
moMG = np.repeat(0.0, 2)
moML = np.repeat(0.0, 2)
Count = OUT[:, 0].size

for i in range(Count):
    wei+=Mexp(0,1,0,OUT[i,2],OUT[i,3])*OUT[i,5]
    for j in range(2):
        moMG[j]=moMG[j]+Mexp(j,1,0,OUT[i,2],OUT[i,3])*OUT[i,5]
        moML[j]=moML[j]+Mexp(0,1,0,OUT[i,2],OUT[i,3])*OUT[i,4]**(j+1)*OUT[i,5]
        
meanG = moMG[0]/wei
sdG = math.sqrt(moMG[1]/wei-meanG**2)
meanL = moML[0]/wei
sdL = math.sqrt(moML[1]/wei-meanL**2)

# Computes transformed means and standard deviations of
# q_G = exp(-lambdaG * xA/N); q_L = exp(-lambdaL)
#

A1 = 1+xA/N
A2 = 1+2*xA/N

WEIq=(np.exp(-OUT[:,2])-np.exp(-OUT[:,3]))*OUT[:,5]
moqG=[]
moqG.append(sum((np.exp(-A1*OUT[:,2])-np.exp(-A1*OUT[:,3]))*OUT[:,5])/A1)
moqG.append(sum((np.exp(-A2*OUT[:,2])-np.exp(-A2*OUT[:,3]))*OUT[:,5])/A2)

moqL=[]
moqL.append(sum(WEIq*np.exp(-OUT[:,4])))
moqL.append(sum(WEIq*np.exp(-2*OUT[:,4])))

meanqG = moqG[0]/wei
sdqG = math.sqrt(moqG[0]/wei-meanqG**2)
meanqL = moqL[0]/wei
sdqL = math.sqrt(moqL[1]/wei-meanqL**2)

# Summarise results
print("Summarise results", simT)

# Parameter means and sd
print("Parameter means and sd", meanG, sdG, meanL, sdL)

# Transformed parameters means and sd (compare with Clancy and O'Neill (2008)
# and Neal (2012))
print("Transformed parameters means and sd", meanqG, sdqG, meanqL, sdqL)

# Time taken.

tendC=time.time()
tendC-tstartC

#####################################################################################################################
#  ABC code for running vanilla ABC
#####################################################################################################################
tstartC = time.time()

DATA=SeattleA


proc = process(DATA)
hsA, isA , colA, rowA,  xA , N = proc

epss = [8,1]
k = 0
run = 10

# Run vanilla ABC
OUTP = House_van(DATA,epss,k,run)
OUT = OUTP['OUTPUT']

# Moment calculations
meanG = OUT[:, 0].mean()
sdG = np.std(OUT[:, 0])
meanL = OUT[:, 1].mean()
sdL = np.std(OUT[:,1])

#transformed moments
meanqG = np.exp(-OUT[:, 0]*xA/N).mean()
sdqG = np.std(np.exp(-OUT[:, 0]*xA/N))
meanqL = np.exp(-OUT[:,1]).mean()
sdqL = np.std(np.exp(-OUT[:,1]))


print('Summarized results',meanG,sdG,meanL,sdL)

# Transformed parameters.
print('Transformed parameters',meanqG,sdqG,meanqL,sdqL)

print('simcount', OUTP['simcount'])

tendC = time.time()
print('running time is', tendC-tstartC)

