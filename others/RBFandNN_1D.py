import numpy as np
import math
import matplotlib.pyplot as plt

#RBF補間の定義--------------------------------------------------------------------

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
#    y = math.exp(-x)
    y = (1-x)**4*(4*x+1)
    return y
    

def test_func(x):
    y = x**2
    return y 

Ns = 90
Na = 30

xs = np.linspace(-5., 5., Ns)
xa = np.linspace(-5., 5., Na)

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
H1 = Hl[:, ndim + 1:]

#--------------------------------------------------------------------------------------

#最近傍補間の定義------------------------------------------------------------------------
H2 =np.zeros((Na, Ns))
for i in range(Ns):
    for j in range(Na):
        if xs[i] - (xs[1]-xs[0])/2 < xa[j] and xa[j] < xs[i] + (xs[1]-xs[0])/2 :
            H2[i, j] +=1
        elif xs[i] + (xs[1]-xs[0])/2 == xa[j]:
            H2[i, j] += 1/2
            H2[i+1, j] += 1/2
        else:


#--------------------------------------------------------------------------------------------

#RBFと最近傍補間の切り替え---------------------------------------------------------------------
H = np.zeros((Na, Ns))
A = []
dc = 0.03

for i in range(Na):
    index = np.abs(xs - xa[i]).argmin()

    if np.linalg.norm(xs[index]-xa[i]) < dc :
        H[i, :] = H2[i, :]
        A += "N"

    else :
        H[i, :] = H1[i, :]
        A += "R"


print(H)
print(A)
#---------------------------------------------------------------------------------------------
#xsからus(変位)を求める
us = test_func(xs)

#ua = H * us
ua = np.dot(H, us.T)

#xfからfaを求める
fa = test_func(xa)

#fs = H.T(Hの転置) * fa
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


if np.dot(us, fs) == np.dot(ua.T, fa.T):
    print(True)

else:
    print(False)
