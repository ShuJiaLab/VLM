import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from os import system
system('cls')

factor = 0.01
maxval = 1
cell_sigma = 64

def destfunc(x,sigma=1):
    return np.exp(((-x**2)/(2*sigma**2))**3)

def rejection_sampling(iter=1000):
    samples = []

    for i in range(iter):
        x = np.random.uniform(-77,77)
        y = np.random.uniform(0, maxval)   

        while y > destfunc(x,cell_sigma):
            x = np.random.uniform(-77,77)
            y = np.random.uniform(0, maxval)
        samples.append(x)

    return np.array(samples)

samples = rejection_sampling(iter=5000)
print(samples.max())
hist, binedges = np.histogram(samples, bins=50, density=True)
plt.bar((binedges[:-1]+binedges[1:])/2.0, hist, width=binedges[1]-binedges[0], color='g', alpha=0.5)
xmin, xmax = (-77,77) #plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = destfunc(x,cell_sigma)
p = p/p.max()*hist[25]
plt.plot(x, p, 'k', linewidth=2)
plt.show()