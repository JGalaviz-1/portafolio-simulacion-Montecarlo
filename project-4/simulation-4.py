import numpy as np
from numba import njit
import time

                                        #PARAMETERS

NOUT = "salida_nn" #name of the output archive
N = 302
R = 2.5
L = 50
l = 6
nmc = 500_000
nb = 200

                                      #FRECUENTLY USED

bines = np.linspace(0, 0.5*L, nb+1) #bins to compute the histogram
vol_bin = (4/3) * np.pi * (bines[1:]**3 - bines[:-1]**3) #volume of each bin

#returns the position (of the sphere with center at y) relative to the one with center at x
#implementing the minimal image convention
@njit
def img(y1,y2,y3, x1,x2,x3, L):
    a1 = y1 - x1 #position y relative to x
    a2 = y2 - x2
    a3 = y3 - x3
    #apply b.c in the box centered at x center
    if a1 < -0.5*L:
        a1 += L
    elif a1 > 0.5*L:
        a1 -= L
    if a2 < -0.5*L:
        a2 += L
    elif a2 > 0.5*L:
        a2 -= L
    if a3 < -0.5*L:
        a3 += L
    elif a3 > 0.5*L:
        a3 -= L

    return a1, a2, a3

#distance from the origin
@njit
def dst(x1,x2,x3):
    return np.sqrt(x1**2 + x2**2 + x3**2)

#returns a vector with the corresponding distances from the central sphere to each of the spheres centers
@njit
def DistCent(cnt):
    n = len(cnt)-1
    cnt_dst = np.zeros(n) #array with the centers distance from (0,0,0)
    for i in range(n): #the distance from all the spheres, except the central, respect the central one is computed
        cnt_dst[i] = dst(cnt[i+1,0],cnt[i+1,1],cnt[i+1,2])
    return cnt_dst

#histogram function
@njit
def histog(cnt_dst,bines):
    nb = len(bines)-1
    h = np.zeros(nb) #histogram array
    for i in range(len(cnt_dst)): #for all spheres centers, except the fixed one
        for j in range(nb): #check in which interval in the bines array is that sphere center
            if bines[j]<=cnt_dst[i]<bines[j+1]:
                h[j] += 1
                break #when is in some bin, dont further check
    return h

#initialization of the sphere centers
@njit
def CNT(N, R, L, l):
    cnt = np.zeros((N,3))  # sphere centers
    # initialize configuration (place spheres without overlap)
    for i in range(1, N): #central sphere is allways at (0,0,0)
        while True: #random positions are generated in the sim. box
            fl = True
            a1 = L*np.random.random() - 0.5*L
            a2 = L*np.random.random() - 0.5*L
            a3 = L*np.random.random() - 0.5*L
            for j in range(i): #check if there are overlaps with the already placed spheres
                #generate the images of the already placed spheres in the simulation box centered at the trial sphere center
                dx,dy,dz = img(cnt[j,0],cnt[j,1],cnt[j,2], a1,a2,a3, L)
                if dst(dx,dy,dz) < 2*R: #hard sphere condition
                    fl = False
                    break
            if fl == True: #if no overlaps, accept
                cnt[i,0] = a1
                cnt[i,1] = a2
                cnt[i,2] = a3
                break
    return cnt

#montecarlo cycle function, modifies the cnt data if a movement is accepted, at the end
#returns the number of accepted moves and the normalized histogram of the radial density function
@njit
def MCC(N, R, L, l,cnt, bines, vol_bin):
    lcl_ac = 0
    for i in range(10):  # 10 montecarlo cycles
        for j in range(N-1): #N-1 movs will be proposed
            z = np.random.randint(1, N) #choose one of the spheres, except the fixed at (0,0,0)
            a1 = cnt[z,0] + l*np.random.random() - 0.5*l #displacing it in the little box, trial coordinate
            a2 = cnt[z,1] + l*np.random.random() - 0.5*l
            a3 = cnt[z,2] + l*np.random.random() - 0.5*l
            fl = True
            for k in range(N):
                #generate the images of the already placed spheres in the simulation box centered at the trial sphere center
                dx,dy,dz = img(cnt[k,0],cnt[k,1],cnt[k,2],a1,a2,a3, L)
                if k != z and dst(dx,dy,dz) < 2*R: #check if the trial sphere position overlaps with any in the sim. box
                    fl = False
                    break
            if fl == True: #if no overlaps then accept the move, if accepted:
                cnt[z,0], cnt[z,1],cnt[z,2] = img(a1,a2,a3,0,0,0, L) #replace the sphere coord. with the trial one
                lcl_ac += 1 #update the acceped movs. counter

    cnt_dst = DistCent(cnt) # distances from particle 0
    hist = histog(cnt_dst,bines) #histogram generation
    rho = hist / vol_bin #normalizing
    return lcl_ac, rho

                                        #MAIN LOOP

@njit
def MAIN(N,nb,nmc, R, L, l, cnt, bines, vol_bin):
    ac = 0 # acum. of total accepted movs.
    exp_rho = np.zeros(nb) # expected density array
    for i in range(int(nmc/10)):
        lcl_ac, rho = MCC(N, R, L, l,cnt, bines, vol_bin)
        ac += lcl_ac #updating ac
        exp_rho += rho / (nmc/10) #updating the average
    g_r = exp_rho * L**3 / N # compute the radial distribution
    return g_r ,ac

t1 = time.time()#time setup
cnt = CNT(N, R, L, l)
g_r,ac = MAIN(N,nb,nmc, R, L, l, cnt, bines, vol_bin)  #g(r) computation
t2 = time.time() #times stops

# Save results
r = 0.5*(bines[1:] + bines[:-1]) #bins midpoint
data = np.column_stack((r, g_r)) #saving as x,y archive
np.savetxt(f"{NOUT}.txt", data)

                                            #OUTPUTS

print(f"acceptance coefficient: {ac/(nmc*(N-1))}")
print(f"time taken: {t2-t1}")

