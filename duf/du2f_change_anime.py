import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#放射基底関数の決め方で振動の挙動が変わる
def func(x):
#    y = math.exp(-x)
    y = (1-x)**4*(4*x+1)
    return y
    
def test_func(x):
    y = x**2
    return y 

Ns = 90
Na = 45

# アニメーション用の描画関数
def update(Ns):
    plt.cla()  # 現在のプロットをクリア
    if Ns < 2:
        return

    xs = np.linspace(-1., 1., Ns)
    xa = np.linspace(-1., 1., Na)

    ndim = 1
    css = np.zeros((len(xs) + ndim + 1, len(xs) + ndim + 1))
    css[ndim + 1:, 0] = 1.
    css[0, ndim + 1:] = 1.
    css[ndim + 1:, 1] = xs.T
    css[1, ndim + 1:] = xs

    css_1 = np.zeros((len(xs), len(xs)))
    for i in range(len(xs)):
        for j in range(len(xs)):
            r1 = np.abs(xs[i] - xs[j])
            css_1[i, j] = func(r1)

    css[ndim + 1:, ndim + 1:] = css_1
    css_inv = np.linalg.inv(css)

    aas = np.zeros((len(xa), len(xs) + ndim + 1))
    aas[:, 0] = 1.
    aas[:, 1] = xa.T

    aas_1 = np.zeros((len(xa), len(xs)))
    for i in range(len(xa)):
        for j in range(len(xs)):
            r2 = np.abs(xa[i] - xs[j])
            aas_1[i, j] = func(r2)

    aas[:, ndim + 1:] = aas_1

    Hl = np.dot(aas, css_inv)
    H = Hl[:, ndim + 1:]

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

    delta_W1 = 0
    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                if abs(xs[i] - xa[j]) <= 1e-8:
                    delta_W1 += us[i] * delta_f[i]
                else:
                    pass
            else:
                pass

#仕事総和が保存するように調整
    delta_W = (us @ (H3 - H.T) @ fa.T) + np.dot(us, delta_f.T) + delta_W1

#duの定義
    du = np.zeros(Na)

#duを加える格子点における流体荷重の総和を計算
    fa_sum = 0
    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                if abs(xs[i] - xa[j]) > 1e-8:
                    fa_sum += fa[j]
                else:
                    pass
            else:
                pass

#Duの計算
    if fa_sum < 1e-10:
        print("Warning: fa_sum = 0")
        sys.exit()
    else:
        Du = delta_W / fa_sum


#座標が一致しない格子点にduを足す
    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                if abs(xs[i] - xa[j]) > 1e-8:
                    #                du[j] = Du
                    ua[j] += Du
                else:
                    pass
            else:
                pass

    

    plt.rcParams["font.size"] = 14
    plt.xlabel("x")    
    plt.ylabel("u")
    plt.plot(xa, ua+du, label='ua+du', marker='o')
    plt.plot(xs, us, label='us', marker='*')
    plt.title(f"Na = {Na}, Ns = {Ns}")
    plt.legend()

# アニメーションの設定
fig = plt.figure(figsize=(9, 5))
ani = animation.FuncAnimation(fig, update, frames=range(2, Ns+1), interval=300)
#ani.save("NNandRBF_usandua_du1.mp4", writer="ffmpeg", fps=2)

plt.show()





















# アニメーション用の描画関数
def update(Ns):
    plt.cla()  # 現在のプロットをクリア
    if Ns < 2:
        return

    xs = np.linspace(-1., 1., Ns)
    xa = np.linspace(-1., 1., Na)

    ndim = 1
    css = np.zeros((len(xs) + ndim + 1, len(xs) + ndim + 1))
    css[ndim + 1:, 0] = 1.
    css[0, ndim + 1:] = 1.
    css[ndim + 1:, 1] = xs.T
    css[1, ndim + 1:] = xs

    css_1 = np.zeros((len(xs), len(xs)))
    for i in range(len(xs)):
        for j in range(len(xs)):
            r1 = np.abs(xs[i] - xs[j])
            css_1[i, j] = func(r1)

    css[ndim + 1:, ndim + 1:] = css_1
    css_inv = np.linalg.inv(css)

    aas = np.zeros((len(xa), len(xs) + ndim + 1))
    aas[:, 0] = 1.
    aas[:, 1] = xa.T

    aas_1 = np.zeros((len(xa), len(xs)))
    for i in range(len(xa)):
        for j in range(len(xs)):
            r2 = np.abs(xa[i] - xs[j])
            aas_1[i, j] = func(r2)

    aas[:, ndim + 1:] = aas_1

    Hl = np.dot(aas, css_inv)
    H = Hl[:, ndim + 1:]


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

    delta_W1 = 0
    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                if abs(xs[i] - xa[j]) <= 1e-8:
                    delta_W1 += us[i] * delta_f[i]
                else:
                    pass
            else:
                pass

    
#仕事総和が保存するように調整
    delta_W = (us @ (H3 - H.T) @ fa.T) + np.dot(us, delta_f.T) + delta_W1

#duの定義
    du = np.zeros(Na)

#duを加える格子点における流体荷重の総和を計算
    fa_sum = 0
    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                if abs(xs[i] - xa[j]) > 1e-8:
                    fa_sum += fa[j]
                else:
                    pass
            else:
                pass

#Duの計算
    if fa_sum < 1e-10:
        print("Warning: fa_sum = 0")
        sys.exit()
    else:
        Du = delta_W / fa_sum


#座標が一致しない格子点にduを足す
    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                if abs(xs[i] - xa[j]) > 1e-8:
                    #                du[j] = Du
                    ua[j] += Du
                else:
                    pass
            else:
                pass

    

    plt.rcParams["font.size"] = 14
    plt.xlabel("x")    
    plt.ylabel("f")
    plt.plot(xa, fa, label='fa', marker='o')
    plt.plot(xs, fs, label='fs', marker='*')
    plt.title(f"Na = {Na}, Ns = {Ns}")
    plt.legend()

# アニメーションの設定
fig = plt.figure(figsize=(9, 5))
ani = animation.FuncAnimation(fig, update, frames=range(2, Ns+1), interval=300)
#ani.save("NNandRBF_fsansfa.mp4", writer="ffmpeg", fps=2)

plt.show()

