import numpy as np
import math
import matplotlib.pyplot as plt

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
#    y = math.exp(-x)
    y = (1-x)**4*(4*x+1)
    return y
    

def test_func(x):
    y = x**2
    return y 

Ns = input("Ns = ")
Na = input("Na = ")

Ns = int(Ns)
Na = int(Na)

xs = np.linspace(-1., 1., Ns)
xa = np.linspace(-1., 1., Na)

us = test_func(xs)
fa = test_func(xa)



#uのRBF補間--------------------------------------------------------------------------
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
aas = np.zeros((len(xa), len(xs) + ndim + 1))

aas[:, 0] = 1.
aas[:, 1] = xa.T

aas_1 = np.zeros((len(xa), len(xs)))
for i in range(len(xa)):
    for j in range(len(xs)):
        r2 = np.linalg.norm(xa[i] - xs[j])
        aas_1[i, j] = func(r2)

aas[:, ndim + 1:] = aas_1

#Hlマトリックスの計算
Hl = np.dot(aas, css_inv)

#Hlからいらない成分を取り出す
H = Hl[:, ndim + 1:]

#----------------------------------------------------------------------------------------




#fのRBF補間------------------------------------------------------------------------------
ndim = 1
css2 = np.zeros((len(xa) + ndim + 1, len(xa) + ndim + 1))
css2[ndim + 1:, 0] = 1.
css2[0, ndim + 1:] = 1.
css2[ndim + 1:, 1] = xa.T
css2[1, ndim + 1:] = xa

css2_1 = np.zeros((len(xa), len(xa)))
for i in range(len(xa)):
    for j in range(len(xa)):
        r3 = np.linalg.norm(xa[i] - xa[j])
        css2_1[i, j] = func(r3)

css2[ndim + 1:, ndim + 1:] = css2_1
css2_inv =  np.linalg.inv(css2)

#aasの定義
aas2 = np.zeros((len(xs), len(xa) + ndim + 1))

aas2[:, 0] = 1.
aas2[:, 1] = xs.T

aas2_1 = np.zeros((len(xs), len(xa)))
for i in range(len(xs)):
    for j in range(len(xa)):
        r4 = np.linalg.norm(xs[i] - xa[j])
        aas2_1[i, j] = func(r4)

aas2[:, ndim + 1:] = aas2_1

#Hlマトリックスの計算
Hl2 = np.dot(aas2, css2_inv)

#Hlからいらない成分を取り出す
H2 = Hl2[:, ndim + 1:]


#---------------------------------------------------------------------------------------




#psiの定義--------------------------------------------------------------------------------
one_a = np.ones(Na)
one_s = np.ones(Ns)

psi = (np.dot(one_a.T, fa.T))/(one_s @ H2 @ fa.T)

#---------------------------------------------------------------------------------------

ua = np.dot(H, us.T)
fs = psi * np.dot(H2, fa.T)

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


#delta_uで仕事が保存するように調整
delta_u = np.dot((H2.T - H),us.T)

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xa, ua, label = 'ua', marker = '*' )
ax.plot(xa, ua.T + delta_u, label = 'ua+delta_u', marker = '+' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'us', marker = 'o' )
ax.plot(xa, ua.T + delta_u, label = 'ua+delta_u', marker = '+' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('delta_u')
ax.plot(xa, delta_u.T, label = 'delta_u', marker = '+' )
ax.legend(fontsize = 16)
plt.show()

