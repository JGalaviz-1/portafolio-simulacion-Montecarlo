

import numpy as np
import matplotlib.pyplot as plt

"""
Box-Mueller method implementation, uses vectorized numpy operations.
This program uses the uniform random numbers array generator of numpy (numpy.random)
which is based on the Mersenne Twister algorithm. Has period of 2**(19937) -1.

Inputs:
-------
N : number of points to generate
Nb: number of bins in which the histogram will be divided
a : lower limit
b : upper limit
NAME : output archive name

Outputs:
--------
resulting mean and variance
plot of the histogram comparing with the normal distribution
text archive with the histogram data corresponding to the bins. For the ith element of the output
vector the following data is stored:
[bin midpoint, normalized frecuency in the bin]
"""

                                           #PARAMETERS

N = 5_000_000 #number of points
Nb = 100 #number of of bins
np.random.seed(None) #seeding the RNG (change None by the seed for comparison)
a,b = -7,7 #limits
NAME = "BM_100" #output archive name


                                       #BM IMPLEMENTATION

#the bins array is generated
bines = np.linspace(a,b,Nb+1)

#the random uniform numbers the BM algorithm uses
u1 = np.random.random(N)
u2 = np.random.random(N)

#implementing the BM algorithm
r = np.sqrt(-2*np.log(u1))
theta = 2*np.pi*u2
X = r*np.cos(theta) #random points array, computed according to BM method

#computation of the mean and variance of the data
mean = sum(X)/N
var = sum(X**2)/N - mean**2

#the histogram is generated using the defined bins
hist, bines = np.histogram(X, bins=bines)

#computation of the area of the histogram
I = 0
D = (b-a)/Nb #histogram bin width
for i in range(Nb):
    I += D*hist[i] #summing the histogram rectangles areas
norm_hist = hist/I #normalization of the histogram

#the program writes a text archive with the bins and bin frecuency (normalized data)
arch = np.array([[(bines[i+1]+bines[i])/2,norm_hist[i]] for i in range(Nb)])
np.savetxt(f"{NAME}.txt", arch)


                                          #PLOTTING

#defining and computing the normal dist. for the array y
def f(x):
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)
y = np.linspace(a,b,200)
F = f(y) #array with the values of the normal curve (mean zero, std = 1)

#plot showing
plt.figure(figsize=(10,5))
plt.bar(bines[:-1],norm_hist,width=D,align="edge",alpha=0.7, label="BM histogram")
plt.plot(y, F, color="orange",
         label=r"normal distribution, $\mu=0, \sigma=1$",
         lw=2)

#displaying the mean and std
texto = f"Generated random\npoints parameters:\n$\\mu$ = {mean:.4f}\n$\\sigma^{2}$ = {var:.4f}"
plt.text(.5*b, .2, texto,
         fontsize=10,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

#plot labels
plt.xlabel("bins")
plt.ylabel("Normalized frecuency in the bin")
plt.title(f"Box-Mueller method results comparision")
plt.legend()
plt.grid()

plt.show()

