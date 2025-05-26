import numpy as np
import matplotlib.pyplot as plt

def f(x):
    y = x**2
    return y

Ns = 30
Na = 78

xs = np.linspace(-5, 5, Ns)
xa = np.linspace(-5, 5, Na)

us = f(xs)
fa = f(xa)


#変位変換行列の定義
H = np.zeros((Na, Ns))
#H[0, 0] = 1
#H[Na-1, Ns-1] = 1

ua = np.zeros(Na)

#Wは補間係数
for i in range(Na):
    for j in range(Ns):
        if xs[j] <= xa[i] and xa[i] <= xs[j+1]:
            W = (xa[i]-xs[j]) / (xs[j+1]-xs[j])
            H[i, j] = 1-W
            H[i, j+1] = W
            break
        else:
            pass

print(H)
print(H.T)

ua = np.dot(H, us.T)
fs = np.dot(H.T, fa.T)


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
