"""
抽象一个用于SVM方法的类
svm支持向量
用于非线性支持向量，包含线性（可分）支持向量
"""
import numpy as np
import random
from svm_util import *


class Svm:

    """
        dataMatIn 输入矩阵
        classLabels 标签矩阵
    """

    def __init__(self, dataMatIn, classLabels, C, toler, ktup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))  # 核函数
        self.k1 = 1.3
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], ktup)

    def __calcEk(self, k):
        fXk = float(np.multiply(self.alphas, self.labelMat).T * self.K[:, k] + self.b)
        Ek = fXk - float(self.labelMat[k])
        return Ek

    # 返回最大步长的j，和Ej
    def __selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(self.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.__calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = select_jrand(i, self.m)
            Ej = self.__calcEk(j)
        return j, Ej

    def __updateEk(self, k):
        Ek = self.__calcEk(k)
        self.eCache[k] = [1, Ek]

    """
    内循环，与上一个版本类似
    特征向量的内积修改为核函数
    """
    def __innerL(self, i):
        Ei = self.__calcEk(i)
        if (self.labelMat[i] * Ei < -self.tol and self.alphas[i] < self.C) or \
                (self.labelMat[i] * Ei > self.tol and self.alphas[i] > 0):
            j, Ej = self.__selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if self.labelMat[i] != self.labelMat[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                # print("L == H")
                return 0
            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                # print("eta>=0")
                return 0
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = clip_alpha(self.alphas[j], H, L)
            self.__updateEk(j)
            # alpha几乎不变
            if abs(self.alphas[j] - alphaJold) < 0.00001:
                # print("j not moving enough")
                return 0
            self.alphas[i] += self.labelMat[j] * self.labelMat[i] * (alphaJold - self.alphas[j])
            self.__updateEk(i)
            b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * \
                 self.K[i, i] - self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[i, j]
            b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * \
                 self.K[i, j] - self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[j, j]
            # 如果alphas[i]和alphas[j]值在0和C之间，则有b1==b2
            if 0 < self.alphas[i] < self.C:
                self.b = b1
            elif 0 < self.alphas[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    # dataMatIn, classLabels, C, toler, maxIter, kTup=("lin", 0)
    def smop(self, maxIter):
        # oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
        iter = 0
        entire_set = True
        alpha_pairs_changed = 0
        while iter < maxIter and (alpha_pairs_changed > 0 or entire_set):
            if entire_set:
                # 遍历所有值
                for i in range(self.m):
                    alpha_pairs_changed += self.__innerL(i)
                    # print("fullSet, iter: %d i:%d. pairs changed %d" %
                    #       (iter, i, alpha_pairs_changed))
                iter += 1
            else:
                # 遍历非边界值
                # 取这两个条件的交集
                non_bound_is = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in non_bound_is:
                    alpha_pairs_changed += self.__innerL(i)
                    # print("non_bound, iter: %d i:%d. pairs changed %d" %
                    #       (iter, i, alpha_pairs_changed))
                iter += 1
            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            # print("iteration number: %d" % iter)
        return self.b, self.alphas


"""
示例
修改以下变量，以期获得最优的训练函数和较低的测试错误率
k1 指的是1.414σ
C 容错率
max_iter 外循环次数
"""


# def test_rbf(max_iter, k1=1.3, C=200):
#     data_arr, label_arr = load_data_set("testSetRBF.txt")
#     svm = Svm(np.mat(data_arr), np.mat(label_arr).transpose(), C, 0.0001, ("rbf", k1))
#     b, alphas = svm.smop(max_iter)
#     data_mat = svm.X
#     label_mat = svm.labelMat
#     sv_ind = np.nonzero(alphas.A > 0)[0]   # alphas是二维，列的值为1
#     svm.sVs = data_mat[sv_ind]
#     svm.label_sv = label_mat[sv_ind]
#     test_train_rbf(svm, data_mat, alphas, label_arr, sv_ind)
#     svm.k1 = k1
#
#     data_arr, label_arr = load_data_set("testSetRBF2.txt")
#     data_mat = np.mat(data_arr)
#     label_mat = np.mat(label_arr).transpose()
#     test_train_rbf(svm, data_mat, alphas, label_arr, sv_ind)
#
#
def test_train_rbf(svm, data_mat, alphas, label_arr, sv_ind):
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(svm.sVs, data_mat[i, :], ("rbf", svm.k1))
        predict = kernel_eval.T * np.multiply(svm.label_sv, alphas[sv_ind]) + svm.b
        if np.sign(predict) != np.sign(label_arr[i]):
            error_count += 1
    print("the training error rate is: %f" % (float(error_count)/m))
#
#
# test_rbf(1000)
