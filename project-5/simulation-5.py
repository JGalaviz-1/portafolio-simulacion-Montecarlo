import numpy as np
from numba import njit
import time

                                        #PARAMETERS

NOUT = "salida_nn"
N = 302
R = 2.5
L = 50
l = 6
nmc = 5000
nb = 200
Q = 1000 #period of each "snapshot", must be multiple of 10

                                      #FRECUENTLY USED

bines = np.linspace(0, 0.5*L, nb+1) #bins to compute the histogram
vol_bin = (4/3) * np.pi * (bines[1:]**3 - bines[:-1]**3) #volume of each bin
pos = np.zeros((int(nmc/Q) + 1,N,3)) #will contain the configurations of each 1000 MC cicles

#returns the position (of the sphere with center at y) relative to the one with center at x
#implementing the minimal image convention
@njit
def img(y1,y2,y3,x1,x2,x3, L):
    a1 = y1 - x1 #position y relative to x
    a2 = y2 - x2
    a3 = y3 - x3
    #apply b.c in the box centered at x center
    if a1 < -0.5*L: a1 += L
    elif a1 > 0.5*L: a1 -= L
    if a2 < -0.5*L: a2 += L
    elif a2 > 0.5*L: a2 -= L
    if a3 < -0.5*L: a3 += L
    elif a3 > 0.5*L: a3 -= L

    return a1,a2,a3

#distance from the origin
@njit
def dst(x1,x2,x3):
    return np.sqrt(x1**2 + x2**2 + x3**2)

#histogram function
@njit
def histog(cnt_dst,bines):
    nb = len(bines)-1
    h = np.zeros(nb) #histogram initialization
    for i in range(len(cnt_dst)): #check in which interval is each distance
        for j in range(nb):
            if bines[j]<=cnt_dst[i]<bines[j+1]: #check if the element i is in the jth bin
                h[j] += 1 #update the histogram
                break
    return h

#computes the array cnt containing all the distances between the particle c
#and the remaining in the simulation box
@njit
def DC(cnt,c,L):
    n = len(cnt)-1 #dont calculate the distance from the sphere c to itself
    cnt_dst = np.zeros(n) # array initialization
    i = 0
    for j in range(len(cnt)): #filling the cnt_dst array
        if j!= c:
            #use minimal image convention and periodic conditions
            #to compute the distances
            dx,dy,dz = img(cnt[j,0],cnt[j,1],cnt[j,2],cnt[c,0],cnt[c,1],cnt[c,2],L)
            cnt_dst[i] = dst(dx,dy,dz)
            i += 1
    return cnt_dst

#initialization of the sphere centers
@njit
def CNT(N, R, L, l):
    cnt = np.zeros((N,3))  # sphere centers
    # initialize configuration (place spheres without overlap)
    for i in range(0, N):
        while True: #random positions are generated in the sim. box
            fl = True #this will track if the proposed sphere overlaps (False) or not (True)
            a1 = L*np.random.rand() - 0.5*L
            a2 = L*np.random.rand() - 0.5*L
            a3 = L*np.random.rand() - 0.5*L
            for j in range(i): #check if there are overlaps with the already placed spheres
                #use minimal image convention and periodicity
                dx,dy,dz = img(cnt[j,0],cnt[j,1],cnt[j,2],a1,a2,a3, L)
                if dst(dx,dy,dz) < 2*R: #then check if the spheres overlap
                    fl = False
                    break
            if fl == True: #if no overlaps, accept
                cnt[i,0] = a1
                cnt[i,1] = a2
                cnt[i,2] = a3
                break
    return cnt

#montecarlo cycles function, modifies the cnt data if a movement is accepted, at the end
#returns the number of accepted moves and the normalized histogram of the radial density function
@njit
def MCC(N, R, L, l,nb,cnt, bines, vol_bin):
    hist = np.zeros(nb) #the histogram for each MC cycle is generated
    lcl_ac = 0 #local acummulator of accepted moves
    for i in range(10):  # 10 montecarlo cycles
        for j in range(N): #a montecarlo cycle
            z = np.random.randint(0, N) #choose one of the spheres, except the fixed at (0,0,0)
            a1 = cnt[z,0] + l*np.random.rand() - 0.5*l #displacing it in the little box, trial coordinate
            a2 = cnt[z,1] + l*np.random.rand() - 0.5*l
            a3 = cnt[z,2] + l*np.random.rand() - 0.5*l
            fl = True #this will track if the proposed sphere overlaps (False) or not (True)
            for k in range(N):
                #use minimal image convention and periodicity to check for overlaps
                dx,dy,dz = img(cnt[k,0],cnt[k,1],cnt[k,2], a1,a2,a3, L)
                if k != z and dst(dx,dy,dz) < 2*R: #check if overlapping occurs for the trial positions a
                    fl = False
                    break
            if fl == True: #if no overlaps then accept the move, if accepted:
                cnt[z,0],cnt[z,1],cnt[z,2] = img(a1,a2,a3,0,0,0, L) #replace the sphere coord. with the trial one
                lcl_ac += 1

    # distances from particle i, then histogram is computed
    # then the N histograms are averaged
    for i in range(N):
        cnt_dst = DC(cnt,i,L) #distances from the ith center
        hist += histog(cnt_dst,bines)/N #histogram generation

    rho = hist / vol_bin #normalizing
    return lcl_ac, rho

                                        #MAIN LOOP

q = int(Q/10)
@njit
def MAIN(N,nb, R, L, l, cnt, bines, vol_bin,pos):
    ac = 0 # acum. of total accepted movs.
    c = 1 #keeps the index in the pos matrix
    exp_rho = np.zeros(nb) # expected density array initialization
    for i in range(int(nmc/10)):
        lcl_ac, rho= MCC(N, R, L, l,nb,cnt, bines, vol_bin)
        ac += lcl_ac #updating ac
        exp_rho += rho / (nmc/10) #updating the average
        if (i+1) %q == 0 : #check is 1000 montecarlo cycles were done
            pos[c,:,:] = cnt #if so, save the configurations in pos matrix
            c += 1
    g_r = exp_rho * L**3 / N # compute the radial distribution
    return g_r ,ac

t1 = time.time()#time setup
cnt = CNT(N, R, L, l) #centers are generated
pos[0,:,:] = cnt #inital configuration is registered
g_r,ac = MAIN(N,nb, R, L, l, cnt, bines, vol_bin,pos)  #g(r) computation
t2 = time.time() #times stops

# Save results
r = 0.5*(bines[1:] + bines[:-1]) #bins midpoint
data = np.column_stack((r, g_r)) #saving as x,y archive
np.savetxt(f"{NOUT}.txt", data)

#configurations are saved as x,y,z archive
with open(f"{NOUT}_cnf.xyz", "w") as f:
    for i in range(int(nmc/Q) + 1):
        f.write(f"{N}\n")                    # number of spheres
        f.write("Hard spheres\n")            # comment line
        for j in range(N):
            f.write(f"X {pos[i,j,0]:.9f} {pos[i,j,1]:.9f} {pos[i,j,2]:.9f}\n")  #sphere coordinates


                                            #OUTPUTS

print(f"acceptance coefficient: {ac/(N*nmc)}")
print(f"time taken: {t2-t1}")

