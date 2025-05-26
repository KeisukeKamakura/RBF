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


#Naの値入力
Na = 10

#Nsの値入力
for Ns in range(2,21):

    xs = np.linspace(-1., 3., Ns)
    xa = np.linspace(-1., 3., Na)


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
            css_inv =  np.linalg.pinv(css)

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

    us = test_func(xs)
    ua = np.dot(H, us.T)

#----------------------------------------------------------------------------------------



#fのRBF補間------------------------------------------------------------------------------
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

#---------------------------------------------------------------------------------------

    fs = psi * np.dot(H2, fa.T)
    H3 = psi * H2


#仕事を保存するようにdelta_uで調整
    delta_u = np.dot((H3.T - H),us.T)

#duの定義
    du = np.zeros(Na)

    fa_sum = 0


    for i in range(Na):
        fa_sum += fa[i]

#Duの計算
    Du = np.dot(delta_u, fa) / fa_sum

    for i in range(Na):
        du[i] = Du

    print('Ns = ', Ns, ', Na = ', Na)

    print('fa_sum = ', np.dot(np.ones(Na), fa))
    print('fs_sum = ', np.dot(np.ones(Ns), fs))
    
    
    print('Wa = ',  np.dot(ua, fa)+np.dot(du, fa))
    print('Ws = ',  np.dot(us, fs))

    print('Ma = ',  np.dot(xa, fa) + np.dot(ua, fa)+ np.dot(du, fa))
    print('Ms = ',  np.dot(xs, fs) + np.dot(us, fs))
    print()
