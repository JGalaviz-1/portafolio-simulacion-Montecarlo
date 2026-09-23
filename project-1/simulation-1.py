

import numpy as np
import time

"""
program implementing random sampling to estimate pi.
the parameter seed is optional (for the RNG to be reproducible)
uses numpy´s uniform random number generator (RNG),having period of recurrence of 2**(19937) -1
and lying between [0,1)

Parameters
----------
N : number of random points to be generated
seed : seed to be used by numpy's random number generator

Outputs
-------
pi_est : estimate of pi
err : relative error (porcentual)
t : time elapsed
"""

#parameters
N = 1_000_000    #number of random points
seed = None       #seed (optional, change None by the seed)


#proposed def. of the computer's precision based value of pi
pi = 6*np.arcsin(1/2)

#the RNG is seeded to compare between approximations of pi
np.random.seed(seed)

t1 = time.time() #cronometer set up

#random points coordinates computation, coordinates between [-1,1)
x = 2*np.random.random(N)-1
y = 2*np.random.random(N)-1

I = 0    #number of points falling on the circle

#array containing the squared distance of the i-th random point to (0,0)
R2 = x**2 + y**2

#counting the number of points inside the unitary circle
#the criterion is: if the distance to (0,0) of the ith point is <=1 (hence the squared is <=1),
#then that point is inside the circle, and then I is incremented by 1
for i in R2:
    if i <= 1:
        I += 1

pi_est = 4*I/N  #resulting approximation of pi
t2 = time.time() #cronometer stops

#percentual relative error got
err = 100*abs((pi_est-pi)/pi)

#printing according to each criterion
print(f"estimado de pi: {pi_est}")
print(f"error relativo: {err} %")
print("tiempo tomado:", t2-t1, "seconds")

