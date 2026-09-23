"""PARALELO """

import numpy as np
from numba import njit,prange,set_num_threads
import time

#for clarity, references are made to the report

a,b = 0,1 #endpoints
ya,yb = 0,-1 #boundary conditions
N = 100 #number of points in the x grid
K = 100_000 #number of random walks beggining at each (grid) point
q = 1 #number of iteration over the approximations
set_num_threads(8) #number of threads in the parallel computation
NAME = "prueba" #name of the output file

#initial guess function
def f(x):
    return (yb-ya)*(x-a)/(b-a) + ya

#functional which is equal to the second derivative of Y in the differential equation
#in order a,b,c represents x,y,dy/dx
@njit
def F(a,b,c):
    return (np.pi**2)*np.sin(np.pi*a)+np.exp(a+np.sin(a*np.pi))-np.exp(-b)

x = np.linspace(a,b,N+1) #gridpoints
Y = f(x) #array with the last approximation of the ODE
h = x[1]-x[0] #interval size

#returns the approximation value at the i-th grid point
#using the last approximation array
@njit(parallel = True)
def Y_C_i(i,K,N,x,y,dy,h):
    yi_app = 0 #stores the accumulated value of V_i,j
               #over all the random walks visited locations
    #generate K random walks
    for j in prange(1,K+1,1):
        z = i #all random walks begin at the i-th grid point
        s = 0 #for the j-th random walk, accumulates the U_i,j value
        #absorbing conditions at the boundaries, set a limit of steps the random walk can have
        while z>0 and z<N:
            #z represents a point in the walk
            a,b,c = x[z],y[z],dy[z] #evaluate the grid value, approximation value of the solution and
                                    #derivative of the approximation at the current point
            s += F(a,b,c) #U_i,j is beeing accumulated
            z = z + 2*np.random.randint(0, 2) - 1 #make a random step (left or right)
        yi_app += y[z]-(h**2)*s/2 #V_i,j is beeing accumulated
    #averaged contributions of all the walks
    return yi_app/K

#yields the next approximation using the last
#one, stores it in the Y array
@njit
def Y_C(K,N,x,Y,h):
    y = Y #using the last approximation
    dy = (Y[1:-1]-Y[0:-2])/h #derivative of the last approximation
    for i in range(1,N,1):
        Y[i] = Y_C_i(i,K,N,x,y,dy,h) #compute the new approximation for each grid point

#iterates over the solutions found to get enhanced results
@njit
def MCS(K,N,x,Y,h,q):
    for i in range(q):
        Y_C(K,N,x,Y,h)

#computation of the solution approximation using the iterative montecarlo
#get the computation time
t1 = time.time()
MCS(K,N,x,Y,h,q)
t2 = time.time()

#save the results
data = np.column_stack((x,Y))
np.savetxt(f"{NAME}.txt", data)

print(f"time taken: {t2-t1} s")