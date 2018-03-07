"""
CHAPTER II - THEORY OF BAYESIAN HAMILTONIAN MONTE CARLO
"""
import matplotlib.pylab as plt
import numpy as np

from scipy.stats import norm
from scipy.special import gamma


"""
LAPLACE APPROXIMATION
"""
def chi2(x, v):
    return x**(v - 1) * np.exp(-(x**2) / 2) / (2**(v/2 - 1) * gamma(v / 2))

k = [2, 10, 25, 30]
x = np.linspace(0, 6, 1000)

for i in k:
    y1 = chi2(x, i)
    y2 = norm.pdf(x, loc = np.sqrt(i - 1), scale = np.sqrt(.5))
    xyplot(x, y1, marker = "", linestyle = "-")
    xyplot(x, y2, marker = "", linestyle = "-", add = True)





