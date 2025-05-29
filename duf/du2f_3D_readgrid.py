import numpy as np
import math
import matplotlib.pyplot as plt
import sys

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
    e = 1
#    y = math.exp(-x)
    y = (1-e*x)**4*(4*e*x+1)
    return y

def test_func(x):
    y = x**2
    return y

#格子点の読み取り-----------------------------------------------------------------

gridstruct = np.loadtxt("file1.txt")

xs = gridstruct[:, 0]
ys = gridstruct[:, 1]
zs = gridstruct[:, 2]
Ns =len(xs)


gridfluid = np.loadtxt("file2.txt")

xa = gridfluid[:, 0]
ya = gridfluid[:, 1]
za = gridfluid[:, 2]
Na =len(xa)

#-------------------------------------------------------------------------------

#構造変位，流体荷重の読み取り-------------------------------------------------------

dispstruct = np.loadtxt("file3.txt")

#us = dispstruct[:, 0]
#vs = dispstruct[:, 1]
#ws = dispstruct[:, 2]


forcefluid = np.loadtxt("file4.txt")

#fxa = forcefluid[:, 0]
#fya = forcefluid[:, 0]
#fza = forcefluid[:, 0]

#-------------------------------------------------------------------------------




#変位のRBF補間--------------------------------------------------------------------------

#cssの定義
ndim = 3
css = np.zeros((len(xs) + ndim + 1, len(xs) + ndim + 1))
css[ndim + 1:, 0] = 1.
css[0, ndim + 1:] = 1.
css[ndim + 1:, 1] = xs.T
css[1, ndim + 1:] = xs
css[ndim + 1:, 2] = ys.T
css[2, ndim + 1:] = ys
css[ndim + 1:, 3] = zs.T
css[3, ndim + 1:] = zs


css_1 = np.zeros((len(xs), len(xs)))
for i in range(len(xs)):
    for j in range(len(xs)):
        r1 = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2 + (zs[i] - zs[j])**2)
        css_1[i, j] = func(r1)

css[ndim + 1:, ndim + 1:] = css_1
css_inv =  np.linalg.pinv(css)

#aasの定義
aas = np.zeros((len(xa), len(xs) + ndim + 1))

aas[:, 0] = 1.
aas[:, 1] = xa.T
aas[:, 2] = ya.T
aas[:, 3] = za.T

aas_1 = np.zeros((len(xa), len(xs)))
for i in range(len(xa)):
    for j in range(len(xs)):
        r2 = math.sqrt((xa[i] - xs[j])**2 + (ya[i] - ys[j])**2 + (za[i] - zs[j])**2)
        aas_1[i, j] = func(r2)

aas[:, ndim + 1:] = aas_1

#Hlマトリックスの計算
Hl = np.dot(aas, css_inv)

#Hlからいらない成分を取り出す
H = Hl[:, ndim + 1:]

#----------------------------------------------------------------------------------------


#荷重のRBF補間←xaとxs変えないといけない---------------------------------------------------------------------------------


#cssの定義
ndim = 3
css = np.zeros((len(xs) + ndim + 1, len(xs) + ndim + 1))
css[ndim + 1:, 0] = 1.
css[0, ndim + 1:] = 1.
css[ndim + 1:, 1] = xs.T
css[1, ndim + 1:] = xs
css[ndim + 1:, 2] = ys.T
css[2, ndim + 1:] = ys
css[ndim + 1:, 3] = zs.T
css[3, ndim + 1:] = zs


css_1 = np.zeros((len(xs), len(xs)))
for i in range(len(xs)):
    for j in range(len(xs)):
        r1 = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2 + (zs[i] - zs[j])**2)
        css_1[i, j] = func(r1)

css[ndim + 1:, ndim + 1:] = css_1
css_inv =  np.linalg.pinv(css)

#aasの定義
aas = np.zeros((len(xa), len(xs) + ndim + 1))

aas[:, 0] = 1.
aas[:, 1] = xa.T
aas[:, 2] = ya.T
aas[:, 3] = za.T

aas_1 = np.zeros((len(xa), len(xs)))
for i in range(len(xa)):
    for j in range(len(xs)):
        r2 = math.sqrt((xa[i] - xs[j])**2 + (ya[i] - ys[j])**2 + (za[i] - zs[j])**2)
        aas_1[i, j] = func(r2)

aas[:, ndim + 1:] = aas_1

#Hlマトリックスの計算
Hl = np.dot(aas, css_inv)

#Hlからいらない成分を取り出す
H = Hl[:, ndim + 1:]




#-----------------------------------------------------------------------------------------



#仕事の総和，モーメントの総和が保存するように調整(x)------------------------------------------------------------

delta_Mx = np.dot(xa, fxa) - np.dot(xs, fxs)

xs2_sum = 0
for i in range(Ns):
    xs2_sum += xs[i]**2

xs_sum = np.sum(xs)


#f = ax + bの係数を求める
A = np.array([[Ns, xs_sum], [xs_sum, xs2_sum]])

B = np.array([0, delta_Mx])

X = np.linalg.solve(A, B)

a = X[1]

b = X[0]


#delta_fを求める
delta_fx = a * xs + b


#仕事総和が保存するように調整
delta_Wx = (us @ (Gx - H.T) @ fxa.T) + np.dot(us, delta_fx.T)

#duの定義
du = np.zeros(Na)

fxa_sum = np.sum(fxa)

#Duの計算
if fxa_sum < 1e-12:
    print("Warning: fxa_sum = 0")
    sys.exit()
else:
    Du = delta_Wx / fxa_sum


for i in range(Na):
    du[i] = Du


#------------------------------------------------------------------------------------------------



#仕事の総和，モーメントの総和が保存するように調整(y)------------------------------------------------------------

delta_My = np.dot(ya, fya) - np.dot(ys, fys)

ys2_sum = 0
for i in range(Ns):
    ys2_sum += ys[i]**2

ys_sum = np.sum(ys)


#fy = cy + dの係数を求める
C = np.array([[Ns, ys_sum], [ys_sum, ys2_sum]])

D = np.array([0, delta_My])

Y = np.linalg.solve(C, D)

c = Y[1]

d = Y[0]


#delta_fを求める
delta_fy = c * ys + d


#仕事総和が保存するように調整
delta_Wy = (vs @ (Gy - H.T) @ fya.T) + np.dot(vs, delta_fy.T)

#duの定義
dv = np.zeros(Na)

fya_sum = np.sum(fya)

#Duの計算
if fya_sum < 1e-12:
    print("Warning: fya_sum = 0")
    sys.exit()
else:
    Dv = delta_Wy / fya_sum


for i in range(Na):
    dv[i] = Dv


#------------------------------------------------------------------------------------------------



#仕事の総和，モーメントの総和が保存するように調整(z)------------------------------------------------------------

delta_Mz = np.dot(za, fza) - np.dot(zs, fzs)

zs2_sum = 0
for i in range(Ns):
    zs2_sum += zs[i]**2

zs_sum = np.sum(zs)


#fz = gz + hの係数を求める
I = np.array([[Ns, zs_sum], [zs_sum, zs2_sum]])

J= np.array([0, delta_Mz])

Z = np.linalg.solve(I, J)

g = Z[1]

h = Z[0]


#delta_fを求める
delta_fz = g * zs + h


#仕事総和が保存するように調整
delta_Wz = (ws @ (Gz - H.T) @ fza.T) + np.dot(ws, delta_fz.T)

#duの定義
dw = np.zeros(Na)

fza_sum = np.sum(fza)

#Duの計算
if fza_sum < 1e-12:
    print("Warning: fza_sum = 0")
    sys.exit()
else:
    Dw = delta_Wz / fza_sum


for i in range(Na):
    dw[i] = Dw


#------------------------------------------------------------------------------------------------



plt.rcParams["font.size"] = 16
fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('x')
ax.set_ylabel('fx')
ax.plot(xs, fxs+delta_fx, label = 'fxs+delta_fx', marker = 'o' )
ax.plot(xa, fxa, label = 'fxa', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'us', marker = 'o' )
ax.plot(xa, ua + du, label = 'ua+du', marker = '+' )
ax.legend(fontsize = 16)
plt.show()


print('Na =',Na , 'Ns =',Ns)
print('fxa_sum = ', np.dot(np.ones(Na), fxa))
print('fxs_sum = ', np.dot(np.ones(Ns), fxs + delta_fx))

print('Wxa = ',  np.dot(ua, fxa) + np.dot(du, fxa))
print('Wxs = ',  np.dot(us, fxs) + np.dot(us, delta_fx))

print('Mxa = ',  np.dot(xa, fxa))
print('Mxs = ',  np.dot(xs, fxs) + np.dot(xs, delta_fx))

print()



plt.rcParams["font.size"] = 16
fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('y')
ax.set_ylabel('fy')
ax.plot(ys, fys+delta_fy, label = 'fys+delta_fy', marker = 'o' )
ax.plot(ya, fya, label = 'fya', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('y')
ax.set_ylabel('v')
ax.plot(ys, vs, label = 'vs', marker = 'o' )
ax.plot(ya, va + dv, label = 'va+dv', marker = '+' )
ax.legend(fontsize = 16)
plt.show()


print('Na =',Na , 'Ns =',Ns)
print('fya_sum = ', np.dot(np.ones(Na), fya))
print('fys_sum = ', np.dot(np.ones(Ns), fys + delta_fy))

print('Wya = ',  np.dot(va, fya) + np.dot(dv, fya))
print('Wys = ',  np.dot(vs, fys) + np.dot(vs, delta_fy))

print('Mya = ',  np.dot(ya, fya))
print('Mys = ',  np.dot(ys, fys) + np.dot(ys, delta_fy))

print()



plt.rcParams["font.size"] = 16
fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('z')
ax.set_ylabel('fz')
ax.plot(zs, fzs+delta_fz, label = 'fzs+delta_fz', marker = 'o' )
ax.plot(za, fza, label = 'fza', marker = '*' )
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_title(f"Na = {Na}, Ns = {Ns}")
ax.set_xlabel('z')
ax.set_ylabel('w')
ax.plot(zs, ws, label = 'zs', marker = 'o' )
ax.plot(za, wa + dw, label = 'wa+dw', marker = '+' )
ax.legend(fontsize = 16)
plt.show()


print('Na =',Na , 'Ns =',Ns)
print('fza_sum = ', np.dot(np.ones(Na), fza))
print('fzs_sum = ', np.dot(np.ones(Ns), fzs + delta_fz)) 

print('Wza = ',  np.dot(wa, fza) + np.dot(dw, fza))
print('Wzs = ',  np.dot(ws, fzs) + np.dot(ws, delta_fz))

print('Mza = ',  np.dot(za, fza))
print('Mzs = ',  np.dot(zs, fzs) + np.dot(zs, delta_fz))

print()
