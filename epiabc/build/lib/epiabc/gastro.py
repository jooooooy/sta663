import numpy

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