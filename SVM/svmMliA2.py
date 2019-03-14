import numpy as np
import random

"""
SMO算法
用于线性（可分）支持向量
"""


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 返回最大步长的j，和Ej
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = select_jrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


"""
i  alpha的下标
m是所有alpha的数目
"""


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


'''
调整alpha 使该值在H和L之间
'''


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or \
            (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        # alpha几乎不变
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 如果alphas[i]和alphas[j]值在0和C之间，则有b1==b2
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smop(dataMatIn, classLabels, C, toler, maxIter, kTup=("lin", 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while iter < maxIter and (alpha_pairs_changed > 0 or entire_set):
        if entire_set:
            # 遍历所有值
            for i in range(oS.m):
                alpha_pairs_changed += innerL(i, oS)
                print("fullSet, iter: %d i:%d. pairs changed %d" %
                      (iter, i, alpha_pairs_changed))
            iter += 1
        else:
            # 遍历非边界值
            # 取这两个条件的交集
            non_bound_is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += innerL(i, oS)
                print("non_bound, iter: %d i:%d. pairs changed %d" %
                      (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def load_data_set(filename):
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split("\t")
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


"""
分类函数
"""


def cal_w(alphas, data_arr, class_labels):
    X = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*label_mat[i], X[i, :].T)
    return w


def cal_y(w, b, x):
    if isinstance(x, list):
        x = np.mat(x)
    y = x*w + b
    return y


data_arr, label_arr = load_data_set("testSet.txt")
b, alphas = smop(data_arr, label_arr, 0.6, 0.001, 40)
w = cal_w(alphas, data_arr, label_arr)
# print(b)
# print(alphas[alphas>0])
print(data_arr)
# print(type(w))

print(cal_y(w, b, data_arr[1]))