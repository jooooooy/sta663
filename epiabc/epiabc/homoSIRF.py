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
