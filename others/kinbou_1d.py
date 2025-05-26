import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
    y = x**2
    return y

Ns = input("Ns = ")
Na = input("Na = ")

Ns = int(Ns)
Na = int(Na)


xs = np.linspace(-1, 1, Ns)
xa = np.linspace(-1, 1 ,Na)

us = f(xs)
fa = f(xa)

H =np.zeros((Ns, Na))
#H[0, 0] = 1
#H[Na-1, Ns-1] = 1


H=np.zeros((Ns, Na))

for i in range(Ns):
    for j in range(Na):
        if xs[i] - (xs[1]-xs[0])/2 < xa[j] and xa[j] < xs[i] + (xs[1]-xs[0])/2 :
            H[i, j] +=1
        elif xs[i] + (xs[1]-xs[0])/2 == xa[j]:
            H[i, j] += 1/2
            H[i+1, j] += 1/2
        else:
            pass


fs = np.dot(H, fa.T)
ua = np.dot(H.T, us.T)

plt.rcParams["font.size"] = 16
fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'us', marker = 'o' )
ax.plot(xa, ua, label = 'ua', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('f')
ax.plot(xs, fs, label = 'fs', marker = 'o' )
ax.plot(xa, fa, label = 'fa', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

