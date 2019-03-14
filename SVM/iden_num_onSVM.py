import numpy as np
from svmMilA4 import Svm
from svmMilA4 import test_train_rbf
from os import listdir
from svm_util import *

"""
手写识别数字
该功能只识别1和9
"""


def img2vector(filename):
    return_vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lin_str = fr.readline()
        for j in range(32):
            return_vect[0, 32*i+j] = int(lin_str[j])
    return return_vect


def load_images(dir_name):
    hwl_labels = []
    training_file_list = listdir(dir_name)
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split(".")[0]
        class_num_str = int(file_str.split("_")[0])
        if class_num_str == 9:
            hwl_labels.append(-1)
        else:
            hwl_labels.append(1)
        training_mat[i, :] = img2vector("%s/%s" % (dir_name, file_name_str))
    return training_mat, hwl_labels


"""
修改以下变量，以期获得最优的训练函数和较低的测试错误率
k1 指的是1.414σ
C 容错率
max_iter 外循环次数
"""


def test_rbf(max_iter, k1=1.3, C=200):
    data_arr, label_arr = load_images("trainingDigits")
    svm = Svm(np.mat(data_arr), np.mat(label_arr).transpose(), C, 0.0001, ("rbf", k1))
    b, alphas = svm.smop(max_iter)
    data_mat = svm.X
    label_mat = svm.labelMat
    sv_ind = np.nonzero(alphas.A > 0)[0]   # alphas是二维，列的值为1
    svm.sVs = data_mat[sv_ind]
    svm.label_sv = label_mat[sv_ind]
    test_train_rbf(svm, data_mat, alphas, label_arr, sv_ind)
    svm.k1 = k1

    data_arr, label_arr = load_data_set("testDigits")
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    test_train_rbf(svm, data_mat, alphas, label_arr, sv_ind)


test_rbf(10, k1=10)
