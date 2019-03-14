import numpy as np


def load_simpdata():
    data_mat = np.matrix([
        [1, 2.1],
        [2, 1.1],
        [1.3, 1],
        [1, 1],
        [2, 1]
    ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


"""
单层决策，小于（或大于某一阈值）进行分类
样本很小，可以全部使用整个数据集
训练完之后，就是对应的G(x)
在训练过程中，要求的对应的tdimen， hresh_val，thresh_ineq
"""

def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == "lt":
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = 1.0
    return ret_array


"""
返回最佳决策树
"""



def build_stump(data_arr, class_labels, D):
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    num_steps = 10.0  # x的个数
    best_stump = {}  # 决策树
    best_class_est = np.mat(np.zeros((m,1)))
    min_error = np.inf
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min)/num_steps
        for j in range(-1, int(num_steps)+1):
            for inequal in ["lt", "gt"]:
                thresh_val = range_min + step_size*float(j)
                predict_vals = stump_classify(
                    data_matrix, i, thresh_val, inequal
                )
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predict_vals == label_mat] = 0
                weighted_error = D.T*err_arr
                # print("split: dim %d, thresh %.2f, thresh inequal: "
                #       "%s, the weighted error is %.3f" %
                #       (i, thresh_val, inequal, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predict_vals.copy()
                    best_stump["dim"] = i  # 第几列特征值
                    best_stump["thresh"] = thresh_val
                    best_stump["ineq"] = inequal
    return best_stump, min_error, best_class_est


"""
基于单层决策树上的分类
可以增加一个迭代次数的参数，用于确保达到某种精度下，不至于运行过多时间
"""


def ada_boost_train(data_arr, class_labels, max_iter=0):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1))/m)
    agg_classest = np.mat(np.zeros((m, 1)))
    iter_num = 0
    while iter_num < max_iter or max_iter == 0:  # 如果增加迭代次数这一参数，修改循环条件为<该迭代次数
        # class_est 返回大于0或小于0的数
        # 需要借助于sign函数 sign(class_est) 就是分类函数G(x),
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        alpha = float(0.5*np.log((1-error)/max(error, 1e-16)))  # 自然对数 防止error过小溢出
        best_stump["alpha"] = alpha
        weak_class_arr.append(best_stump)
        # print("class_est:", class_est.T)
        # 正确分类是 -alpha（符号相同） 否则就是alpha
        #  这样写更简便，不用在使用if-else判断
        expon = np.multiply(-1*alpha*np.mat(class_labels).T, class_est)
        # 不断修改样本的权重值D
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        agg_classest += alpha*class_est
        # print("agg_classest:   ", agg_classest.T)
        # 矩阵中得到的值 1-正确分类的值， -1 错误分类的值
        agg_errors = np.multiply(
            np.sign(agg_classest) != np.mat(class_labels).T,
            np.ones((m,1))
        )
        error_rate = agg_errors.sum()/m
        print("total_errors ", error_rate)
        if error_rate == 0.0:  # 训练错误率为0时退出
            break
        iter_num += 1
    return weak_class_arr, agg_classest


def ada_classify(dat_to_class, classifier_arr):
    data_matrix = np.mat(dat_to_class)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        # classifier_arr 可以用一个实体class封装
        class_est = stump_classify(data_matrix, classifier_arr[i]["dim"],
                    classifier_arr[i]["thresh"], classifier_arr[i]["ineq"],
                    )
        # print("class_est的size:  ",np.shape(class_est))
        agg_class_est += classifier_arr[i]["alpha"]*class_est
    return np.sign(agg_class_est)


def ada_classify_nosign(dat_to_class, classifier_arr):
    data_matrix = np.mat(dat_to_class)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        # classifier_arr 可以用一个实体class封装
        class_est = stump_classify(data_matrix, classifier_arr[i]["dim"],
                    classifier_arr[i]["thresh"], classifier_arr[i]["ineq"],
                    )
        # print("class_est的size:  ",np.shape(class_est))
        agg_class_est += classifier_arr[i]["alpha"]*class_est
    return agg_class_est


# # D = np.mat(np.ones((5, 1))/5.0)
# data_mat, class_labels = load_simpdata()
# # best_stump, min_error, best_class_est = build_stump(data_mat, class_labels, D)
# weak_class_arr = ada_boost_train(data_mat, class_labels)
# # 输出结果：
# # [
# # {'dim': 0, 'thresh': 1.3, 'ineq': 'lt', 'alpha': 0.6931471805599453},
# # {'dim': 1, 'thresh': 1.0, 'ineq': 'lt', 'alpha': 0.9729550745276565},
# # {'dim': 0, 'thresh': 0.9, 'ineq': 'lt', 'alpha': 0.8958797346140273}
# # ]
#
# weak_class_arr =  [
#         {'dim': 0, 'thresh': 1.3, 'ineq': 'lt', 'alpha': 0.6931471805599453},
#         {'dim': 1, 'thresh': 1.0, 'ineq': 'lt', 'alpha': 0.9729550745276565},
#         {'dim': 0, 'thresh': 0.9, 'ineq': 'lt', 'alpha': 0.8958797346140273}
#     ]
# dat_to_class = [
#     [5,2],
#     [0,0],
#     [2,3]
# ]
# ada_classify(dat_to_class, weak_class_arr)
