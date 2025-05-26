import numpy as np
import math
import cmath
import matplotlib.pyplot as plt

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
    y = (1-x)**4*(4*x+1)
    return y
    

def test_func(x):
    y = x**2
    return y

Ns = 60
Na = 30

xs = np.linspace(-1., 1., Ns)
xa = np.linspace(-1., 1., Na)

#cssの定義
ndim = 1
css = np.zeros((len(xs) + ndim + 1, len(xs) + ndim + 1))
css[ndim + 1:, 0] = 1.
css[0, ndim + 1:] = 1.
css[ndim + 1:, 1] = xs.T
css[1, ndim + 1:] = xs

phi_ss = np.zeros((len(xs), len(xs)))
for i in range(len(xs)):
    for j in range(len(xs)):
        r1 = np.linalg.norm(xs[i] - xs[j])
        phi_ss[i, j] = func(r1)

css[ndim + 1:, ndim + 1:] = phi_ss
css_inv =  np.linalg.inv(css)

#aasの定義
aas = np.zeros((len(xa), len(xs) + ndim + 1))

aas[:, 0] = 1.
aas[:, 1] = xa.T

phi_as = np.zeros((len(xa), len(xs)))
for i in range(len(xa)):
    for j in range(len(xs)):
        r2 = np.linalg.norm(xa[i] - xs[j])
        phi_as[i, j] = func(r2)

aas[:, ndim + 1:] = phi_as

#Hlマトリックスの計算
Hl = np.dot(aas, css_inv)

#Hlからいらない成分を取り出す
H = Hl[:, ndim + 1:]

#xsからus(変位)を求める
us = test_func(xs)

#ua = H * us
ua = np.dot(H, us.T)

#xfからfaを求める
fa = test_func(xa)

#fs = H.T(Hの転置) * fa
fs = np.dot(H.T, fa.T)

###############################################################
#F(ローパスフィルター)の定義
#F_1 = np.zeros((Ns-1 , Ns-1), dtype = complex )
#for a in range(Ns-1):
#    for b in range(Ns-1):
#        F_1[a, b] = math.e**(-1j*2*math.pi*(a+1)*(b+1)/Ns)

###############################################################

#fsにフィルターかける→fs_filter

cut_wave = 4
#座標を周波数領域に
F_2 = np.fft.fftfreq(xs.shape[0], xs[1] - xs[0])
#fsをフーリエ変換
X_F = np.fft.fft(fs)

#排除する周波数
X_F[F_2 > cut_wave] = 0
X_F[F_2 < -cut_wave] = 0
#X_Fを逆フーリエ変換し，実部を抽出
fs_filter =np.fft.ifft(X_F).real

##################################################################
#F = np.zeros((Ns, Ns), dtype = complex)
#F[:, 0] = 1
#F[0, 1:] = 1
#F[1:, 1:] = F_2

#print(F)

#C_cutの定義
#C_cut = np.zeros((Ns, Ns))
#for i in range(10):
#    C_cut[i, i] = 1
#    C_cut[Ns-1-i, Ns-1-i] = 1

#Qの計算
#Q = np.fft.ifft(C_cut @ F_2).real
#print(Q)


#fs_filter = np.dot(X_CF, fs)
#print(fs_filter)

#ガウシアンフィルターの定義
#sigma_2 = 300
#r = np.exp(-1 / (2 * sigma_2))
#xr = np.zeros(Ns)
#xr[30] = 1
#for i in range(1, 30):
 #   xr[30-i] = r**(i**2)
  #  xr[30+i] = r**(i**2)

#print(xr)


#fs_filter = np.zeros(Ns)
#for i in range(Ns):
#    fs_filter[i] = xr[i] * fs[i]

#########################################################################








#フィルターかけた後の変位．荷重変換行列の再定義
#aasの拡張
Aas = np.zeros((len(xa) + 1, len(xs) + ndim + 1))
Aas[1:, :] = aas
#a_A1
Aas[0, 0] = 0
#a_A2
Aas[0, 1] = 0
#aの定義
a = np.dot(phi_ss, fs_filter) - np.dot(phi_as.T, fa.T)

Aas[0, 2:] = a.T

#a_u
c6 = css_inv[2:, 2:]
a_u = a.T @ c6 @ us.T

#a_f
a_f = 1

#新しい変位変換行列W
Wl = np.dot(Aas, css_inv)
W = Wl[:, ndim + 1:]

#Ua
ua_change = np.zeros(Na)
fa_sum = 0
for i in range(Na):
    fa_sum += fa[i]
du = a_u * a_f / fa_sum

for j in range(Na):
    ua_change[j] = ua[j] + du

Ua = np.zeros((Na+1, 1))
Ua[0, 0] = a_u
Ua[1:, 0] = ua_change

#Us
Us = np.zeros((Ns+2, 1))
Us[0, 0] = 0
Us[1, 0] = 0
Us[2:, 0] = us.T

#Fs
Fs = np.zeros((Ns+2, 1))
Fs[0, 0] = 0
Fs[1, 0] = 0
Fs[2:, 0] = fs_filter

#Fa
Fa = np.zeros((Na+1, 1))
Fa[0, 0] = a_f
Fa[1:, 0] = fa.T

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(xs, us, label = 'dispstruct', marker = 'o' )
ax.plot(xa, ua, label = 'dispfluid', marker = '*' )
ax.plot(xa, ua_change, label = 'dispfluid_change', marker = '^') 
ax.legend(fontsize = 16)
plt.show()

fig = plt.figure(figsize = (8, 5))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('f')
ax.plot(xs, fs, label = 'forcestruct', marker = 'o' )
ax.plot(xa, fa, label = 'forcefluid', marker = '*' )
ax.plot(xs, fs_filter, label = 'forcestruct_filter', marker = '^' )
ax.legend(fontsize = 16)
plt.show()

#変位変換行列の行の和(荷重変換行列の列の和)
sum = np.zeros((Na, 1))
for i in range(Na):
    for j in range(Ns):
        sum[i, 0] += W[i, j]

print(sum)
print(W)
