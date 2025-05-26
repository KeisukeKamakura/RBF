import numpy as np
import math
import matplotlib.pyplot as plt
import sys

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
    e = 1
#    y = math.exp(-(e*x))
    y = (1-e*x)**4*(4*e*x+1)
    return y

def test_func(x):
#    y = x + 1
    y = x**2
    return y

Ns = input("Ns = ")
Na = input("Na = ")

Ns = int(Ns)
Na = int(Na)

xs = np.linspace(-1., 1., Ns)
xa = np.linspace(-1., 1., Na)

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

us = test_func(xs)
ua = np.dot(H, us.T)

#fのRBF補間---------------------------------------------------------------------------------

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

fa = test_func(xa)

#psiの定義
one_a = np.ones(Na)
one_s = np.ones(Ns)

psi = (np.dot(one_a.T, fa.T))/(one_s @ H2 @ fa.T)

fs = psi * np.dot(H2, fa.T)
H3 = psi * H2

#モーメントの総和が保存するように調整------------------------------------------------------------

delta_M = np.dot(xa, fa) - np.dot(xs, fs)

xs2_sum = 0
for i in range(Ns):
    xs2_sum += xs[i]**2

xs_sum = np.sum(xs)

#----------------------------------------------------------------------------------------------------------------------------

#f = ax + bの係数を求める
A = np.array([[Ns, xs_sum], [xs_sum, xs2_sum]])

B = np.array([0, delta_M])

X = np.linalg.solve(A, B)

a = X[1]

b = X[0]


#f = ax の係数aを求める

#xs2_sum = 0
#for i in range(Ns):
#    xs2_sum += xs[i]**2

#a = delta_M / xs2_sum

#-----------------------------------------------------------------------------------------------------------------------------

#delta_fを求める---------------------
delta_f = a * xs + b


#仕事総和が保存するように調整
delta_W = (us @ (H3 - H.T) @ fa.T) + np.dot(us, delta_f.T)

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
    Du = delta_W / fa_sum


for i in range(Na):
    du[i] = Du

plt.rcParams["font.size"] = 16
fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'us', marker = 'o' )
ax.plot(xa, ua + du, label = 'ua + du', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('x')
ax.set_ylabel('f')
ax.plot(xs, fs+delta_f, label = 'fs+delta_f', marker = 'o' )
ax.plot(xa, fa, label = 'fa', marker = '*' )
ax.legend(fontsize = 16)
plt.show()



print('Na =', Na , 'Ns =', Ns)
print('fa_sum = ', np.dot(np.ones(Na), fa))
print('fs_sum = ', np.dot(np.ones(Ns), fs))
    

print('Wa = ',  np.dot(ua, fa) + np.dot(du, fa))
print('Ws = ',  np.dot(us, fs) + np.dot(us, delta_f))

print('Ma = ',  np.dot(xa, fa))
print('Ms = ',  np.dot(xs, fs) + np.dot(xs, delta_f))

#print()
#print('delta_f = ', delta_f)
#print('du =', du)


