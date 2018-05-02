import numpy as np
import math
import time

Tecumseh1 = np.zeros((8,7))
Tecumseh1[0,0:5] = np.array((66 ,87 ,  25  , 22   , 4 ))
Tecumseh1[1,0:5] = np.array((13 ,14 ,  15  ,  9   , 4 ))
Tecumseh1[2,1:6] = np.array((4 ,   4  ,  9  ,  2 ,   1 ))
Tecumseh1[3,2:7] = np.array(( 4   , 3   , 1  ,  1 ,   1))
Tecumseh1[4,3:5] = np.array(( 1,1))
Tecumseh1


def thresH(n):
    thres = np.repeat(0.0, n - 1)
    thres[0] = np.random.exponential(size=1, scale=1 / (n - 1))
    if n > 2:
        for i in range(2, n):
            thres[i - 1] = thres[i - 2] + np.random.exponential(size=1, scale=1 / (n - i))

    return thres


def process(DATA):
    hsA = np.sum(DATA, axis=0)
    isA = np.sum(DATA, axis=1)
    colA = np.array(range(1, len(hsA) + 1))
    rowA = np.array(range(0, len(hsA) + 1))

    xA = np.sum(isA * rowA)  # Final size
    N = np.sum(hsA * colA)  # Population size

    return {'hsA': hsA, 'isA': isA, 'colA': colA, 'rowA': rowA, 'xA': xA, 'N': N}


def House_epi(n, k, lambda_L):
    i = 0
    sev = 0

    if n == 1:
        i = 1
        if k == 0:
            sev = 1
        if k > 0:
            sev = np.random.gamma(k, 1 / k, 1)

    if n > 1:
        t = thresH(n)
        if k == 0:
            q = np.repeat(1.0, n)
        if k > 0:
            q = np.random.gamma(size=n, shape=k, scale=1 / k)
        t = np.append(t, 2 * lambda_L * np.sum(q))

        i = 0
        test = 0
        while test == 0:
            i = i + 1
            if t[i - 1] > (lambda_L * np.sum(q[0:i])):
                test = 1
                sev = np.sum(q[0:i])

    return np.array([i, sev])


def House_SEL(hs, lambda_G, lambda_L, k):
    HH = hs.size
    ks = np.array(range(1, HH + 1))

    n = np.repeat(ks, hs)

    m = n.size  # Number of households
    N = n.sum()  # Population size
    NS = N

    sev = 0
    threshold = 0

    ni = np.repeat(0, m)
    ns = n.copy()

    R = np.random.exponential(size=N)

    while threshold <= lambda_G * sev:
        kk = np.random.choice(a=np.arange(m) + 1, size=1, replace=True, p=ns / (ns.sum()))[0]

        hou_epi = House_epi(ns[kk - 1], k, lambda_L)
        ns[kk - 1] = ns[kk - 1] - hou_epi[0]
        sev = sev + hou_epi[1]

        ni[kk - 1] = n[kk - 1] - ns[kk - 1]
        # NS=NS-hou_epi[0]
        NS = ns.sum()

        if NS > 0:
            threshold = threshold + np.random.exponential(scale=N / NS, size=1)[0]
        if NS == 0:
            threshold = 2 * lambda_G * sev

    OUT = np.zeros((HH + 1, HH))
    for i in range(1, HH + 2):
        for j in range(1, HH + 1):
            OUT[i - 1, j - 1] = sum(n[ni == (i - 1)] == j)

    return OUT


def House_van(Xdata, epsil, k, run):
    """
    Xdata - a data set
    epsil - a numpy array of length 2
    k - int
    run - int
    """

    OUTPUT = np.zeros((run, 2))
    hs = np.sum(Xdata, axis=0).astype('int')
    isB = np.sum(Xdata, axis=1)

    rowB = np.arange(hs.size + 1)
    xB = sum(isB * rowB)

    simcount = 0
    j = 0

    while j < run:
        simcount = simcount + 1
        lambda_G = np.random.exponential(size=1)
        lambda_L = np.random.exponential(size=1)
        J = House_SEL(hs, lambda_G, lambda_L, k)

        if ((abs(J - Xdata)).sum()) <= epsil[0]:
            isJ = np.sum(J, axis=1)
            if abs(sum(isJ * rowB) - xB) <= epsil[1]:
                j = j + 1
                OUTPUT[j - 1,] = np.array([lambda_G, lambda_L]).reshape((1, 2))
                print(j, simcount)

    return {'OUTPUT': OUTPUT, 'simcount': simcount}


def House_imp(Xdata, epsil, k, run, OX):
    """Code for running ABC for households with importance sampling"""
    OUTPUT = np.zeros((run, 3))
    hs = np.sum(Xdata, axis=0).astype('int')
    isB = np.sum(Xdata, axis=1).astype('int')
    rowB = np.arange(hs.shape[0] + 1)
    xB = np.sum(np.multiply(isB, rowB))

    simcount, j = 0, 0
    # Compute variances for the pertubations. Bivariate Gaussian
    meanG = np.sum(np.multiply(OX[:, 0], OX[:, 2])) / np.sum(OX[:, 2])
    meanL = np.sum(np.multiply(OX[:, 1], OX[:, 2])) / np.sum(OX[:, 2])
    vG = np.sum(np.multiply(OX[:, 0] ** 2, OX[:, 2])) / np.sum(OX[:, 2]) - meanG ** 2
    vL = np.sum(np.multiply(OX[:, 1] ** 2, OX[:, 2])) / np.sum(OX[:, 2]) - meanL ** 2
    vLG = np.sum(OX[:, 0] * OX[:, 1] * OX[:, 2]) / np.sum(OX[:, 2]) - meanG * meanL
    vaR = 2 * np.array([vG, vLG, vLG, vL]).reshape((2, 2)).T
    Sinv = np.linalg.inv(vaR)
    sG = math.sqrt(2 * vG)
    sLL = math.sqrt((vL - vLG ** 2 / vG) * 2)
    sAL = vLG / vG

    while j < run:
        # print(simcount)
        simcount += 1
        LA = np.random.choice(run, 1, p=OX[:, 2] / OX[:, 2].sum(), replace=True)[0]
        lambda_G = np.random.normal(OX[LA - 1, 0], sG, 1)[0]
        lambda_L = np.random.normal((OX[LA - 1, 1] + sAL * (lambda_G - OX[LA - 1, 0])), sLL, 1)[0]

        if lambda_G > 0 and lambda_L > 0:
            J = House_SEL(hs, lambda_G, lambda_L, k)
            if np.sum(abs(J - Xdata)) <= epsil[0]:
                isJ = np.sum(J, axis=1).astype('int')
                # if (abs(sum(isJ * rowB) - xB) <= epsil[2])
                if abs(sum(isJ * rowB) - xB) <= epsil[1]:
                    j += 1
                    weiZ = 0
                    for i in range(1, run + 1):
                        xDIF = np.array([lambda_G, lambda_L]) - OX[LA - 1, 0:2]
                        mult = xDIF.T @ Sinv @ xDIF
                        weiZ = weiZ + np.exp(-mult / 2)
                    weiG = np.exp(-1 * (lambda_L + lambda_G)) / weiZ
                    OUTPUT[j - 1, :] = [lambda_G, lambda_L, weiG]
                    print(j, simcount)

    return {'OUTPUT': OUTPUT, 'simcount': simcount}





DATA = Tecumseh1

# Main code for partially coupled ABC for household epidemics
tstartC = time.time()

# Process data
proc = process(DATA)
hsA = proc['hsA']
isA = proc['isA']
colA = proc['colA']
rowA = proc['rowA']
xA = proc['xA']
N = proc['N']

# Code for running coupled algorithm and selecting which simulations to keep.
# Set thresholds

Eps1=np.array((100,70,50))
Eps2=np.array((10,6,4))


TT = Eps1.size
EpsM = (np.r_[Eps1,Eps2]).reshape((-1,2),order='F')

k = 0
run = 500

epss = EpsM[0,]


start = time.time()
np.random.seed(123)

OUTP = House_van(DATA,epss,k,run)
OUT = OUTP['OUTPUT']
simT = OUTP['simcount']
OL = np.c_[OUT, np.repeat(1,run)]

if TT > 1:
    for t in range(2,TT+1):
        OUTP = House_imp(DATA, EpsM[t-1,], k, run, OL)
        OL = OUTP['OUTPUT']
        simT = simT + OUTP['simcount']

OUT = OL
end = time.time()
# Moment calculations for SMC-ABC output

weiA = sum(OUT[:,2])
meanG = sum(OUT[:,0]*OUT[:,2])/weiA
sdG = np.sqrt(sum(OUT[:,0]**2 * OUT[:,2])/weiA-meanG**2)
meanL = sum(OUT[:,1]*OUT[:,2])/weiA
sdL = np.sqrt(sum(OUT[:,1]**2 * OUT[:,2])/weiA-meanL**2)


# Moment calculations for transformed variables

meanqG = sum(np.exp(-OUT[:,0] * xA/N) * OUT[:,2])/weiA
sdqG = np.sqrt(sum(np.exp(-2 * OUT[:,0] * xA/N) * OUT[:,2])/weiA-meanqG**2)
meanqL = sum(np.exp(-OUT[:,1])*OUT[:,2])/weiA
sdqL = np.sqrt(sum(np.exp(-2 * OUT[:,1]) * OUT[:,2])/weiA - meanqL**2)


# Summarise results
print(np.array((meanG,sdG,meanL,sdL)))

# Transformed parameters.
print(np.array((meanqG,sdqG,meanqL,sdqL)))

print(simT)

duration = np.array((end-start,))

print(duration)

np.savetxt('C:\\Users\\jings\\Dropbox\\Spring 2018\\663\\FINAL PROJECT\\data\\time_pmc.txt', duration)
np.savetxt('C:\\Users\\jings\\Dropbox\\Spring 2018\\663\\FINAL PROJECT\\data\\post_pmc.txt', OUT)