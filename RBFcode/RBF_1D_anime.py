import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 放射基底関数
def func(x):
    return (1 - x)**4 * (4 * x + 1)

# 元の関数
def test_func(x):
    return x**2


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

    plt.rcParams["font.size"] = 14
    plt.xlabel("x")    
    plt.ylabel("u")
    plt.plot(xa, ua, label='ua', marker='o')
    plt.plot(xs, us, label='us', marker='*')
    plt.title(f"Na = {Na}, Ns = {Ns}")
    plt.legend()

# アニメーションの設定
fig = plt.figure(figsize=(9, 5))
ani = animation.FuncAnimation(fig, update, frames=range(2, 61), interval=200)

plt.show()

def update2(Ns):
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

    fa = test_func(xa)
    fs = np.dot(H.T, fa.T)

    plt.rcParams["font.size"] = 14
    plt.xlabel("x")    
    plt.ylabel("f")
    plt.plot(xa, fa, label='fa', marker='*')
    plt.plot(xs, fs, label='fs', marker='o')
    plt.title(f"Na = {Na}, Ns = {Ns}")
    plt.legend()

# アニメーションの設定
fig = plt.figure(figsize=(9, 5))
ani = animation.FuncAnimation(fig, update2, frames=range(2, 61), interval=200)

plt.show()


#plt.rcParams["font.size"] = 16
#fig = plt.figure(figsize = (8, 5))
#ax = fig.add_subplot(111)
#ax.set_xlabel('x')
#ax.set_ylabel('u')
#ax.plot(xs, us, label = 'us', marker = 'o' )
#ax.plot(xa, ua, label = 'ua', marker = '*' )
#plt.show()
