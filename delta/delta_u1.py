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

#RBF補間--------------------------------------------------------------------------

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



#最近傍補間G---------------------------------------------------------------------------------
G =np.zeros((Ns, Na))

for i in range(Ns):
    for j in range(Na):
        if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
            G[i, j] = 1
            break
        else:
            pass

#-----------------------------------------------------------------------------------------

us = test_func(xs)
ua = np.dot(H, us.T)

fa = test_func(xa)
fs = np.dot(G, fa.T)

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

#仕事を保存するようにdelta_uで調整
delta_u = np.dot((G.T - H),us.T)

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


if np.dot(us, fs) == np.dot(ua, fa) + np.dot(delta_u, fa):
    print(True)

else:
    print(False)








