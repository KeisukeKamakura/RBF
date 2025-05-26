import numpy as np
import math
import matplotlib.pyplot as plt
from RBFclass import RBF1D

def wendlands_c2(x):
    y = (1 - x)**4 * (4 * x + 1)
    return y 
        
def test_func(x):
    y = x**2
    return y

Ns = input("Ns = ")
Na = input("Na = ")

Ns = int(Ns)
Na = int(Na)

ndim = 1
xs = np.linspace(-1., 1., Ns)
xa = np.linspace(-1., 1., Na)

RBF_1D = RBF1D(ndim, wendlands_c2, xs, xa)

#これしないとH使えない
RBF_1D.RBF1D_calculate()

H = RBF_1D.H

us = test_func(xs)
ua = np.dot(H, us.T)

fa = test_func(xa)
fs = np.dot(H.T, fa.T)

plt.rcParams["font.size"] = 16
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label='us', marker='o')
ax.plot(xa, ua, label='ua', marker='*')
ax.legend(fontsize=16)
plt.show()

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('f')
ax.plot(xs, fs, label='fs', marker='o')
ax.plot(xa, fa, label='fa', marker='*')
ax.legend(fontsize=16)
plt.show()
