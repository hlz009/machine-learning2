import numpy as np
from adaboost import ada_boost_train, ada_classify, ada_classify_nosign

def load_dataset(filename):
    data_mat = []
    label_mat = []
    lines = open(filename).readlines()
    for line in lines:
        line_arr = []
        curr_line = line.strip().split("\t")
        for num in curr_line:
            line_arr.append(float(num))
        data_mat.append(line_arr)
        label_mat.append(float(curr_line[-1]))
    return data_mat, label_mat


# data_arr, label_arr = load_dataset("horseColicTraining2.txt")
# classifier, agg_classest = ada_boost_train(data_arr, label_arr)
# test_data_arr, test_label_arr = load_dataset("horseColicTest2.txt")
# predict = ada_classify(test_data_arr,classifier)
#
# # print(np.shape(predict))
# error_arr = np.mat(np.ones((67, 1)))
# error_rate = error_arr[predict != np.mat(test_label_arr).T].sum()
# print(error_rate)