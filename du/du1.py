import numpy as np
import math
import matplotlib.pyplot as plt
import sys

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
        if xs[i] - (xs[1]-xs[0])/2 < xa[j] and xa[j] < xs[i] + (xs[1]-xs[0])/2 :
            G[i, j] +=1
#        elif abs(xs[i] + (xs[1]-xs[0])/2 - xa[j]) < 1e-12:
#            G[i, j] += 1/2
#            if i + 1 < Ns:
#                G[i+1, j] += 1/2
        else:
            pass


#-----------------------------------------------------------------------------------------


fa = test_func(xa)
fs = np.dot(G, fa.T)

us = test_func(xs)
ua = np.dot(H, us.T)

#仕事を保存するようにdelta_uで調整
delta_u = np.dot((G.T - H),us.T)

#duの定義
du = np.zeros(Na)

fa_sum = 0

for i in range(Na):
    fa_sum += fa[i]

#Duの計算
if fa_sum < 1e-12:
    print("Warning: fa_sum = 0")
    sys.exit()
else:
    Du = np.dot(delta_u, fa) / fa_sum

for i in range(Na):
    du[i] = Du

plt.rcParams["font.size"] = 16
fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'us', marker = 'o' )
ax.plot(xa, ua + du, label = 'ua+du', marker = '+' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('x')
ax.set_ylabel('f')
ax.plot(xs, fs, label = 'fs', marker = 'o' )
ax.plot(xa, fa, label = 'fa', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

print('Na =',Na ,'Ns =',Ns)
print('fa_sum = ', np.dot(np.ones(Na), fa))
print('fs_sum = ', np.dot(np.ones(Ns), fs))

print('Wa = ',  np.dot(ua, fa)+np.dot(du, fa))
print('Ws = ',  np.dot(us, fs))

print('Ma = ',  np.dot(xa, fa))
print('Ms = ',  np.dot(xs, fs))




