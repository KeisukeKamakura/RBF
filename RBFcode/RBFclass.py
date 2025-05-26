import numpy as np
import math
import matplotlib.pyplot as plt

#以下のRBFはusからuaの変換で使える

class RBF1D:

    def __init__(self, ndim, func_basis, xs, xa):
        self.ndim = ndim
        #self.radius入れるかどうか
        self.func_basis = func_basis
        self.xs = xs
        self.xa = xa
        self.H = None

        return

    def RBF1D_calculate(self):

        # cssの定義
        css = np.zeros((len(self.xs) + self.ndim + 1, len(self.xs) + self.ndim + 1))
        css[self.ndim + 1:, 0] = 1.
        css[0, self.ndim + 1:] = 1.
        css[self.ndim + 1:, 1] = self.xs.T
        css[1, self.ndim + 1:] = self.xs

        css_1 = np.zeros((len(self.xs), len(self.xs)))
        for i in range(len(self.xs)):
            for j in range(len(self.xs)):
                r1 = np.linalg.norm(self.xs[i] - self.xs[j])
                css_1[i, j] = self.func_basis(r1)  

        css[self.ndim + 1:, self.ndim + 1:] = css_1
        css_inv = np.linalg.inv(css)

        # aasの定義
        aas = np.zeros((len(self.xa), len(self.xs) + self.ndim + 1))

        aas[:, 0] = 1.
        aas[:, 1] = self.xa.T

        aas_1 = np.zeros((len(self.xa), len(self.xs)))
        for i in range(len(self.xa)):
            for j in range(len(self.xs)):
                r2 = np.linalg.norm(self.xa[i] - self.xs[j])
                aas_1[i, j] = self.func_basis(r2)  

        aas[:, self.ndim + 1:] = aas_1

        # Hlマトリックスの計算
        Hl = np.dot(aas, css_inv)

        # Hlからいらない成分を取り出す
        self.H = Hl[:, self.ndim + 1:]  # 計算した H をクラスの属性として保存

        return

class RBF2D:

    def __init__(self, ndim, func_basis, xs, xa, ys, ya):
        self.ndim = ndim
        self.func_basis = func_basis
        self.xs = xs
        self.xa = xa
        self.ys = ys
        self.ya = ya
        self.H = None

        return

    def RBF2D_calculate(self):

        # cssの定義
        css = np.zeros((len(self.xs) + self.ndim + 1, len(self.xs) + self.ndim + 1))
        css[self.ndim + 1:, 0] = 1.
        css[0, self.ndim + 1:] = 1.
        css[self.ndim + 1:, 1] = self.xs.T
        css[1, self.ndim + 1:] = self.xs
        css[self.ndim + 1: , 2] = self.ys.T
        css[2, self.ndim + 1:] = self.ys

        css_1 = np.zeros((len(self.xs), len(self.xs)))
        for i in range(len(self.xs)):
            for j in range(len(self.xs)):
                r1 = math.sqrt((self.xs[i] - self.xs[j])**2 + (self.ys[i] - self.ys[j])**2) 
                css_1[i, j] = self.func_basis(r1)  

        css[self.ndim + 1:, self.ndim + 1:] = css_1
        css_inv = np.linalg.inv(css)

        # aasの定義
        aas = np.zeros((len(self.xa), len(self.xs) + self.ndim + 1))

        aas[:, 0] = 1.
        aas[:, 1] = self.xa.T
        ass[:, 2] = self.ya.T

        aas_1 = np.zeros((len(self.xa), len(self.xs)))
        for i in range(len(self.xa)):
            for j in range(len(self.xs)):
                r2 = math.sqrt((xa[i] - xs[j])**2 + (ya[i] - ys[j])**2)
                aas_1[i, j] = self.func_basis(r2)  

        aas[:, self.ndim + 1:] = aas_1

        # Hlマトリックスの計算
        Hl = np.dot(aas, css_inv)

        # Hlからいらない成分を取り出す
        self.H = Hl[:, self.ndim + 1:]  # 計算した H をクラスの属性として保存

        return


class RBF3D:

    def __init__(self, ndim, func_basis, xs, xa, ys, ya, zs, za):
        self.ndim = ndim
        self.func_basis = func_basis
        self.xs = xs
        self.xa = xa
        self.ys = ys
        self.ya = ya
        self.zs = zs
        self.za = za
        self.H = None

        return

    def RBF3D_calculate(self):

        # cssの定義
        css = np.zeros((len(self.xs) + self.ndim + 1, len(self.xs) + self.ndim + 1))
        css[self.ndim + 1:, 0] = 1.
        css[0, self.ndim + 1:] = 1.
        css[self.ndim + 1:, 1] = self.xs.T
        css[1, self.ndim + 1:] = self.xs
        css[self.ndim + 1: , 2] = self.ys.T
        css[2, self.ndim + 1:] = self.ys
        css[self.ndim + 1: , 3] = self.zs.T
        css[3, self.ndim + 1:] = self.zs


        css_1 = np.zeros((len(self.xs), len(self.xs)))
        for i in range(len(self.xs)):
            for j in range(len(self.xs)):
                r1 = math.sqrt((self.xs[i] - self.xs[j])**2 + (self.ys[i] - self.ys[j])**2 + (self.zs[i] - self.zs[j])**2) 
                css_1[i, j] = self.func_basis(r1)  

        css[self.ndim + 1:, self.ndim + 1:] = css_1
        css_inv = np.linalg.inv(css)

        # aasの定義
        aas = np.zeros((len(self.xa), len(self.xs) + self.ndim + 1))

        aas[:, 0] = 1.
        aas[:, 1] = self.xa.T
        ass[:, 2] = self.ya.T
        ass[:, 3] = self.za.T

        aas_1 = np.zeros((len(self.xa), len(self.xs)))
        for i in range(len(self.xa)):
            for j in range(len(self.xs)):
                r2 = math.sqrt((xa[i] - xs[j])**2 + (ya[i] - ys[j])**2 + (za[i] - zs[j])**2)
                aas_1[i, j] = self.func_basis(r2)  

        aas[:, self.ndim + 1:] = aas_1

        # Hlマトリックスの計算
        Hl = np.dot(aas, css_inv)

        # Hlからいらない成分を取り出す
        self.H = Hl[:, self.ndim + 1:]  # 計算した H をクラスの属性として保存

        return


