"""
svm支持向量
用于非线性支持向量，包含线性（可分）支持向量
"""
import numpy as np
import random


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, ktup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))  #核函数
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], ktup)


"""
径向基核函数（高斯函数）
xi --- 全部的特征向量
x --- 指定的特征向量
ktup 第一个参数是指定核函数类型
"""


def kernel_trans(xi, x, ktup):
    m, n = np.shape(xi)
    k = np.mat(np.zeros((m, 1)))
    if ktup[0] == "lin":
        """
        线性核函数（线性支持向量中使用）
        """
        k = xi*x.T
    elif ktup[0] == "rbf":
        """
        高斯核函数
        ktup 第二个参数，1.414σ
        """
        for j in range(m):
            # print(xi[j, :], x)
            delta_row = xi[j, :] - x
            k[j] = delta_row*delta_row.T
        # 除法
        k = np.exp(k / (-1*ktup[1]**2))
    else:
        # 抛出异常
        raise NameError("Knernel is not recongnized")
    return k


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
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


"""
内循环，与上一个版本类似
特征向量的内积修改为核函数
"""


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
            # print("L == H")
            return 0
        eta = 2.0*oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            # print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        # alpha几乎不变
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            # print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
             oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
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
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while iter < maxIter and (alpha_pairs_changed > 0 or entire_set):
        if entire_set:
            # 遍历所有值
            for i in range(oS.m):
                alpha_pairs_changed += innerL(i, oS)
                # print("fullSet, iter: %d i:%d. pairs changed %d" %
                #       (iter, i, alpha_pairs_changed))
            iter += 1
        else:
            # 遍历非边界值
            # 取这两个条件的交集
            non_bound_is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += innerL(i, oS)
                # print("non_bound, iter: %d i:%d. pairs changed %d" %
                #       (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        # print("iteration number: %d" % iter)
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


# def cal_w(alphas, data_arr, class_labels):
#     X = np.mat(data_arr)
#     label_mat = np.mat(class_labels).T
#     m, n = np.shape(X)
#     w = np.zeros((n, 1))
#     for i in range(m):
#         w += np.multiply(alphas[i]*label_mat[i], X[i, :].T)
#     return w
#
#
# def cal_y(w, b, x):
#     if isinstance(x, list):
#         x = np.mat(x)
#     y = x*w + b
#     return y


"""
k1 指的是1.414σ
"""


def test_rbf(k1 = 1.3):
    data_arr, label_arr = load_data_set("testSetRBF.txt")
    b, alphas = smop(data_arr, label_arr, 200, 0.0001, 1000, ("rbf", k1))
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    print(np.shape(alphas.A))
    sv_ind = np.nonzero(alphas.A > 0)[0]   # alphas是二维，列的值为1
    sVs = data_mat[sv_ind]
    label_sv = label_mat[sv_ind]
    # print(np.shape(sVs), np.shape(data_mat))
    # print(sv_ind)
    # print("there are %d support vectors" % np.shape(sVs)[0])
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(sVs, data_mat[i, :], ("rbf", k1))
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
        if np.sign(predict) != np.sign(label_arr[i]):
            error_count += 1
    print("the training error rate is: %f" % (float(error_count)/m))

    data_arr, label_arr = load_data_set("testSetRBF2.txt")
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(sVs, data_mat[i, :], ("rbf", k1))
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
        if np.sign(predict) != np.sign(label_arr[i]):
            error_count += 1
    print("the training error rate is: %f" % (float(error_count) / m))


test_rbf()
