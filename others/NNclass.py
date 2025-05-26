import numpy as np
import math
import matplotlib.pyplot as plt


class NN1D:

    def __init__(self, xs, xa):
        self.xs = xs
        self.xa = xa
    














def f(x):
    y = x**2
    return y

Ns = 60
Na = 30

xs = np.linspace(-1, 1, Ns)
xa = np.linspace(-1, 1 ,Na)

us = f(xs)
fa = f(xa)

H =np.zeros((Ns, Na))
#H[0, 0] = 1
#H[Na-1, Ns-1] = 1

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

print(H)
print(H.T)

#変位変換行列の行の和(荷重変換行列の列の和)
a = np.zeros((Na, 1))
for i in range(Na):
    for j in range(Ns):
        a[i, 0] += H[i, j]

print(a)


if np.dot(us, fs) == np.dot(ua.T, fa.T):
    print(True)

else:
    print(False)
