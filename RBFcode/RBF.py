import numpy as np
import math
import matplotlib.pyplot as plt

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
    y = math.exp(-x)
    return y
    

def test_func(x):
    y = x**2
    return y 

Ns = 40
Nf = 50

xs = np.linspace(-1., 1., Ns)
xf = np.linspace(-1., 1., Nf)

#cssの定義
ndim = 1
css = np.zeros((len(xs) + ndim + 1, len(xs) + ndim + 1))
css[ndim + 1:, 0] = 1.
css[0, ndim + 1:] = 1.
css[ndim + 1:, 1] = xs.T
css[1, ndim + 1:] = xs

css_1 = np.zeros((len(xs), len(xs)))
for i in range(len(xs)):
    for j in range(len(xs)):
        r1 = np.linalg.norm(xs[i] - xs[j])
        css_1[i, j] = func(r1)

css[ndim + 1:, ndim + 1:] = css_1
css_inv =  np.linalg.inv(css)

#aasの定義
aas = np.zeros((len(xf), len(xs) + ndim + 1))

aas[:, 0] = 1.
aas[:, 1] = xf.T

aas_1 = np.zeros((len(xf), len(xs)))
for i in range(len(xf)):
    for j in range(len(xs)):
        r2 = np.linalg.norm(xf[i] - xs[j])
        aas_1[i, j] = func(r2)

aas[:, ndim + 1:] = aas_1

#Hlマトリックスの計算
Hl = np.dot(aas, css_inv)

#Hlからいらない成分を取り出す
H = Hl[:, ndim + 1:]

#xsからus(変位)を求める
us = test_func(xs)

#ua = H * us
ua = np.dot(H, us.T)

#xfからfaを求める
fa = test_func(xf)

#fs = H.T(Hの転置) * fa
fs = np.dot(H.T, fa.T)

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
ax.set_xlabel('x')
ax.set_ylabel('f')
ax.plot(xs, fs, label = 'forcestruct', marker = 'o' )
ax.plot(xf, fa, label = 'forcefluid', marker = '*' )
ax.legend(fontsize = 16)
plt.show()
