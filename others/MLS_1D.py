import numpy as np
import math
import matplotlib.pyplot as plt

#重み基準半径
r0 = 0.2

def func(x):
    y = x**2
    return y

#重み係数
def func_w(r):
    if 0<=r and r<=r0:
        return 1 - 6*(r/r0)**2 + 8*(r/r0)**3 - 3*(r/r0)**4
    if r > r0:
        return 0

Ns = 60
Na = 30

xs = np.linspace(-1, 1, Ns)
xa = np.linspace(-1, 1, Na)

us = func(xs)
ua = np.zeros(Na)

fa = func(xa)

#行列Pの定義
P = np.ones((Ns, 2))

for j in range(Ns):
    P[j, 1] = xs[j]

#行列Wの定義
W = np.zeros((Ns, Ns))

#rの定義
r = np.zeros((1, Ns))

#Gの定義
G = np.zeros((Ns, Na))

for i in range(Na):
#[]は0からスタート, P_xi
    P_xi_T = [1 ,xa[i]]

    for j in range(Ns):
        r[0, j] = np.linalg.norm(xa[i] - xs[j])
        W[j, j] = func_w(r[0, j])

#G_Tの定義
    G_T = P_xi_T @ np.linalg.pinv(P.T @ W @ P) @ P.T @ W
    
#G_Tとusの内積
    ui = np.dot(G_T, us.T)
    ua[i] = ui

#Gの定義
    G[:, i] = G_T
    
fs = np.dot(G, fa.T)

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'dispstruct', marker = 'o' )
ax.plot(xa, ua, label = 'dispfluid', marker = '*' )
ax.legend(fontsize = 16)
plt.show()
    
fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('f')
ax.plot(xs, fs, label = 'forcestruct', marker = 'o' )
ax.plot(xa, fa, label = 'forcefluid', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

#変位変換行列の行の和(荷重変換行列の列の和)
a = np.zeros((Na, 1))
for i in range(Na):
    for j in range(Ns):
        a[i, 0] += G.T[i, j]

print(a)
