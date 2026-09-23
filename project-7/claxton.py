"""CLAXTON-BENSON """

import numpy as np
from numba import njit
import time

                                                            #PARAMETERS

N = 2 #number of particles
w = 800 #number of CLaxton-Benson iterations
K = .15
                                                          #FRECUENTLY USED

#dictionary with the "exact" values of the Thomson problem, for N = 2,3,4,10,20,40,80,100,1000,10000
ex = {2:.5,3:np.sqrt(3),4:3.674234614174767,10:32.7169494,20:150.8815683,40:660.6752788,80:2805.3558759,100:4448.3506343,1000:482535.0,10000:4.94474e7}


#computes the distance between two points
#point 1 with x coords. and point 2 with y coords.
@njit
def dst(x1,x2,x3,y1,y2,y3):
    return np.sqrt((x1-y1)**2+(x2-y2)**2+(x3-y3)**2)


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


#computes the electrostatic energy given the positions of the particles "crd"
@njit
def f_u(crd):
    u = 0 #initialize
    #this is the sum over non repeated particles distances
    for i in range(len(crd)):
        for j in range(i+1,len(crd)):
            #actualize the energy
            u += 1/dst(crd[i,0],crd[i,1],crd[i,2],crd[j,0],crd[j,1],crd[j,2])
    return u


#returns the force components of the force that the k-th particle feels
#is going to be used when determining the new coordinate of the particle
@njit
def F_vect(k,crd):
    F1,F2,F3 = 0,0,0 #force comp. initialization
    #the force is vector components is computed as in the
    #usual electrostatic way
    for i in range(len(crd)):
        if i !=k : #no self forces
            #compute the force components
            d3 = dst(crd[k,0],crd[k,1],crd[k,2],crd[i,0],crd[i,1],crd[i,2])**3
            F1 += (crd[k,0]-crd[i,0])/d3
            F2 += (crd[k,1]-crd[i,1])/d3
            F3 += (crd[k,2]-crd[i,2])/d3
    return F1,F2,F3

#out of the force felt by each particle in the sphere, this function
#determines what is the maximum value in this set
@njit
def F_max(crd):
    #initializa the maximum force as one of the particles´felt force modulo
    F1,F2,F3 = F_vect(0,crd)
    F = np.sqrt(F1**2+F2**2+F3**2)
    #check for the remainig particles, if is bigger actualize
    for i in range(1,len(crd)):
        F1,F2,F3 = F_vect(i,crd)
        F_trial = np.sqrt(F1**2+F2**2+F3**2)
        if F_trial<F:
            F = F_trial
    return F

#claxton-benson method implementation wrapper
@njit
def CB(crd):
    for i in range(w): # w cycles will be performed
        # this value is the prescripted in the method
        # K is chosen in the beggining
        gamma = K/F_max(crd)
        for j in range(N): # each cycle moves all the particles
            #compute the force felt by each particle
            F1,F2,F3 = F_vect(j,crd)
            #move it according to the method and normalize the position
            l = np.sqrt((crd[j,0] + gamma*F1)**2 + (crd[j,1] + gamma*F2)**2 + (crd[j,2] + gamma*F3)**2)
            crd[j,0] = (crd[j,0] + gamma*F1)/l
            crd[j,1] = (crd[j,1] + gamma*F2)/l
            crd[j,2] = (crd[j,2] + gamma*F3)/l



                                                     #METHOD IMPLEMENTATION

coord = f_coord(N) #initialize the coordinates of the particles in the unit sphere
t1 = time.time() #to determine the time taken
CB(coord) #perform the method
t2 = time.time()

                                                            #RESULTS

u_obt = f_u(coord) #resulting electrostatic energy
print(f"time taken: {t2-t1}")
print(f"final energy: {u_obt}")

#save the coordinates as x,y,z archive
if N!= 10000:
    with open(f"coord_C_{N}.txt", "w") as f:
        f.write(f"{N}\n")
        f.write(f"electrostatic energy: {u_obt}\n")
        for i in range(N):
            f.write(f"X {coord[i,0]} {coord[i,1]} {coord[i,2]}\n")

exacto = ex[N]
print(f"RESULTADO EXACTO: {exacto}")
print(f"ERROR PORCENTUAL = {100*np.abs((exacto-u_obt)/exacto)}")

