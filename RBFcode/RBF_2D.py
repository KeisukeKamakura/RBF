import numpy as np
import math
import matplotlib.pyplot as plt

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
    y = (1-x**4)*(4*x+1)
    return y


def f(x):
    z = x**2
    return z

def g(y):
    z = y**4 
    return z

Ns = 60
Nf = 30

#xsとxf,ysとyfは同じ定義域にする
xs = np.linspace(-1., 1., Ns)
ys = np.linspace(-2., 2., Ns)
xf = np.linspace(-1., 1., Nf)
yf = np.linspace(-2., 2., Nf)

#Cssの定義
ndim = 2
css = np.zeros((len(xs) + ndim + 1, len(xs) + ndim + 1))
css[ndim + 1:, 0] = 1.
css[0, ndim + 1:] = 1.
css[ndim + 1:, 1] = xs.T
css[1, ndim + 1:] = xs
css[ndim + 1: , 2] = ys.T
css[2, ndim + 1:] = ys

css_1 = np.zeros((len(xs), len(xs)))
for i in range(len(xs)):
    for j in range(len(xs)):
        r1 = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2) 
        css_1[i, j] = func(r1)

css[ndim + 1:, ndim + 1:] = css_1
css_inv =  np.linalg.pinv(css)

#aasの定義
aas = np.zeros((len(xf), len(xs) + ndim + 1))

aas[:, 0] = 1.
aas[:, 1] = xf.T
aas[:, 2] = yf.T

aas_1 = np.zeros((len(xf), len(xs)))
for i in range(len(xf)):
    for j in range(len(xs)):
        r2 = math.sqrt((xf[i] - xs[j])**2 + (yf[i] - ys[j])**2)
        aas_1[i, j] = func(r2)

aas[:, ndim + 1:] = aas_1

#Hlマトリックスの計算
Hl = np.dot(aas, css_inv)

#Hlからいらない成分を取り出す
H = Hl[:, ndim + 1:]

#xsからus(変位)を求める
us = f(xs)
vs = g(ys)

#ua = H * us
ua = np.dot(H, us.T)
va = np.dot(H, vs.T)

#xfからfaを求める
fxa = f(xf)
fya = g(yf)

#fs = H.T(Hの転置) * fa
fxs = np.dot(H.T, fxa.T)
fys = np.dot(H.T, fya.T)

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'dispstruct', marker = 'o' )
ax.plot(xf, ua, label = 'dispfluid', marker = '*' )
ax.legend(fontsize = 16)
plt.show()



fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('y')
ax.set_ylabel('v')
ax.plot(ys, vs, label = 'forcestruct', marker = 'o' )
ax.plot(yf, va, label = 'forcefluid', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('fx')
ax.plot(xs, fxs, label = 'dispstruct', marker = 'o' )
ax.plot(xf, fxa, label = 'dispfluid', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('y')
ax.set_ylabel('fy')
ax.plot(ys, fys, label = 'dispstruct', marker = 'o' )
ax.plot(yf, fya, label = 'dispfluid', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

#変位変換行列の行の和(荷重変換行列の列の和)
a = np.zeros((Nf, 1))
for i in range(Nf):
    for j in range(Ns):
        a[i, 0] += H[i, j]


#2次元変位変換行列
G = np.zeros((2*Nf, 2*Ns))
G[:Nf, :Ns] = H
G[Nf:, Ns:] = H
print(G)
