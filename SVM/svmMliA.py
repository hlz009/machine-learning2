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
    i  alpha的下标
   m是所有alpha的数目
'''


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


'''
调整alpha 使该值在H和L之间
'''


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


'''
@dataMat    ：数据列表
@classLabels：标签列表
@C          ：权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）
@toler      ：容错率
@maxIter    ：最大迭代次数
'''


def smo_simple(data_matrix_in, class_labels, C, toler, max_iter):
    data_matrix = np.mat(data_matrix_in)
    label_mat = np.mat(class_labels).T
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            # 预测值
            fXi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            Ei = fXi - float(label_mat[i])
            if (label_mat[i] * Ei < -toler and alphas[i] < C) or (label_mat[i] * Ei > toler and alphas[i] > 0):
                j = select_jrand(i, m)  # 随机选出第一个alpha的下标
                fXj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:  # yi!=yj
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    #                     print("L == H")
                    continue
                # K11+K22-2K12
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T \
                      - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    #                     print("eta >= 0")
                    continue
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alpha_j_old) < 0.00001:  # 修改值达到0.00001
                    #                     print("j not moving enough")
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[i] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
        #                 print("iter: %d i: %d, pairs changed %d" % (iter, i , alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
    #         print("iteration number: %d" % iter)
    return b, alphas


data_arr, label_arr = load_data_set("testSet.txt")
b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
# # print(np.shape(alphas[alphas>0]))
print(b)
