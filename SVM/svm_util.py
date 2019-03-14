import random
import numpy as np


def load_data_set(filename):
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split("\t")
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


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
            delta_row = xi[j, :] - x
            k[j] = delta_row*delta_row.T
        # 除法
        k = np.exp(k / (-1*ktup[1]**2))
    else:
        # 抛出异常
        raise NameError("Knernel is not recongnized")
    return k


"""
i  alpha的下标
m是所有alpha的数目
"""


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j