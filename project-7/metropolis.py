"""METROPOLIS """

import numpy as np
from numba import njit
import time
                                                            #PARAMETERS
N = 3 #number of particles
n = 600 #number of montecarlo cycles per temperature
m = 300 #number of temperature steps
T = 1e-4 #maximum temperature
L = .5 #trial cube lenght
                                                          #FRECUENTLY USED

#dictionary with the "exact" values of the Thomson problem, for N = 2,3,4,10,20,40,80,100,1000,10000
ex = {2:.5,3:np.sqrt(3),4:3.674234614174767,10:32.7169494,20:150.8815683,40:660.6752788,80:2805.3558759,100:4448.3506343,1000:482535.0,10000:4.94474e7}

#computes the distance between two points
#point 1 with x coords. and point 2 with y coords.
@njit
def dst(x1,x2,x3,y1,y2,y3):
    return np.sqrt((x1-y1)**2+(x2-y2)**2+(x3-y3)**2)

#computes the electrostatic energy given the positions of the particles "crd"
#this is used for small number of particles
@njit
def f_u(crd):
    u = 0 #initialize
    #this is the sum over non repeated particles distances
    for i in range(len(crd)):
        for j in range(i+1,len(crd)):
            #actualize the energy
            u += 1/dst(crd[i,0],crd[i,1],crd[i,2],crd[j,0],crd[j,1],crd[j,2])
    return u

#computes the energy of the k-th particle with x positions respect the remaining
@njit
def f_eng(x1,x2,x3,k,crd):
    eng = 0 #initialization
    #loop over all distances of the k-th particle with all the remaining points
    for i in range(len(crd)):
        if i!=k :
            eng +=1/dst(x1,x2,x3,crd[i,0],crd[i,1],crd[i,2])  #actualize the energy
    return eng

#generates an array of random points in the unit sphere
@njit
def f_coord(N):
    crd = np.zeros((N,3)) #array initialization
    for i in range(N):
        #random coordinates
        x1 = 2*np.random.rand()-1
        x2 = 2*np.random.rand()-1
        x3 = 2*np.random.rand()-1
        #normalize so it is in the unit sphere
        l = np.sqrt(x1**2+x2**2+x3**2)
        crd[i,0],crd[i,1],crd[i,2] = x1/l,x2/l,x3/l #save it
    return crd

#a Metropolis-Montecarlo simulation wrapper
@njit
def MCS(crd,N,T,m,n,L):
    #perform n*m Metropolis-Montecarlo cycles
    for i in range(m,0,-1): #correspondint to m work temperatures
        for j in range(n): #each temperature will perform n Metropolis-Montecarlo cycles
            for k in range(N): #definition of a Montecarlo cycle
                z = np.random.randint(0,N-1) #select a random point
                #move it randomly inside a cube of lenght L centered at the point
                #so a trial position is generated
                a1 = crd[z,0] + L*np.random.rand()- (.5*L)
                a2 = crd[z,1] + L*np.random.rand()- (.5*L)
                a3 = crd[z,2] + L*np.random.rand()- (.5*L)
                #normalize the trial position
                l = np.sqrt(a1**2 + a2**2 + a3**2)
                a1,a2,a3 = a1/l , a2/l , a3/l
                #if the electrostatic energy of the trial position is less
                #than was before then accept it
                #and replace the old position with the trial one
                du = f_eng(a1,a2,a3,z,crd)-f_eng(crd[z,0],crd[z,1],crd[z,2],z,crd)
                if du <= 0:
                    crd[z,0],crd[z,1],crd[z,2] = a1,a2,a3
                #if the energy is not minimized, then accept it with
                #probability given by the Boltzmann factor
                else:
                    if np.random.rand() < np.exp(-du/(T*i/m)):
                        crd[z,0],crd[z,1],crd[z,2] = a1,a2,a3


                                                            #SIMULATION

coord = f_coord(N) #initialize the coordinates of the particles in the unit sphere
t1 = time.time() #to determine the time taken
MCS(coord,N,T,m,n,L) #perform the simulation
t2 = time.time()

                                                            #RESULTS

u_obt = f_u(coord) #resulting electrostatic energy
np.savetxt(f"TBU_{N}.txt",coord) #is going to be used later on
print(f"time taken: {t2-t1} s")
print(f"final configuration energy: {u_obt}")

#save the coordinates as x,y,z archive
if N != 10_000:
    with open(f"coord_Metro_{N}.txt", "w") as f:
        f.write(f"{N}\n")
        f.write(f"electrostatic energy: {u_obt}\n")
        for i in range(N):
            f.write(f"X {coord[i,0]} {coord[i,1]} {coord[i,2]}\n")

exacto = ex[N]
print(f"RESULTADO EXACTO: {exacto}")
print(f"ERROR PORCENTUAL: {100*np.abs((exacto-u_obt)/exacto)}")

