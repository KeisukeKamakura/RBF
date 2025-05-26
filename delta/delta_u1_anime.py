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

Ns = 60
Na = 30

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

    G =np.zeros((Ns, Na))

    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                G[i, j] = 1
                break
            else:
                pass

    #仕事を保存するようにdelta_uで調整
    delta_u = np.dot((G.T - H),us.T)

    plt.rcParams["font.size"] = 14
    plt.xlabel("x")    
    plt.ylabel("delta_u")
    plt.plot(xa, delta_u, marker='o')
    plt.title(f"Na = {Na}, Ns = {Ns}")
    plt.legend()

# アニメーションの設定
fig = plt.figure(figsize=(9, 5))
ani = animation.FuncAnimation(fig, update, frames=range(2, Ns+1), interval=500)
#ani.save("NNandRBF_delta_u.mp4", writer="ffmpeg", fps=2)

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

    G =np.zeros((Ns, Na))

    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                G[i, j] = 1
                break
            else:
                pass

    #仕事を保存するようにdelta_uで調整
    delta_u = np.dot((G.T - H),us.T)

    plt.rcParams["font.size"] = 14
    plt.xlabel("x")    
    plt.ylabel("u")
    plt.plot(xa, ua+delta_u, label='ua+delta_u', marker='o')
    plt.plot(xs, us, label='us', marker='*')
    plt.title(f"Na = {Na}, Ns = {Ns}")
    plt.legend()

# アニメーションの設定
fig = plt.figure(figsize=(9, 5))
ani = animation.FuncAnimation(fig, update, frames=range(2, Ns+1), interval=300)
#ani.save("NNandRBF_usandua.mp4", writer="ffmpeg", fps=2)

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

    G =np.zeros((Ns, Na))

    for i in range(Ns):
        for j in range(Na):
            if xs[i] - (xs[1]-xs[0])/2 <= xa[j] and xa[j] <= xs[i] + (xs[1]-xs[0])/2 :
                G[i, j] = 1
                break
            else:
                pass

    fa = test_func(xa)
    fs = np.dot(G, fa.T)

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




