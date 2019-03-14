import numpy as np
import matplotlib.pyplot as plt
import adaboost_demo as ad
from adaboost import ada_boost_train, ada_classify, ada_classify_nosign


def plot_ROC(pred_strengths, class_labels):
    cur = (1.0, 1.0)
    y_sum = 0.0
    num_pos_clas = sum(np.array(class_labels) == 1.0)
    y_step = 1/float(num_pos_clas)  # 阳
    x_step = 1/float(len(class_labels) - num_pos_clas)  # 阴
    sorted_indicies = pred_strengths.argsort()
    print(pred_strengths)
    print(class_labels)
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        # ax.plot([cur[0], cur[0]-del_x], [cur[1], cur[1]-del_y], c='r')
        ax.plot([cur[0], cur[0]-del_x], [cur[1], cur[1]-del_y], c='r')
        cur = (cur[0]-del_x, cur[1]-del_y)
    ax.plot([0,1], [0,1], "b--")
    plt.xlabel("False Positive for AdaBoost Horse Colic Detection System")
    ax.axis([-2,2,-2,2])
    plt.show()
    print("the area under the curve is:", y_sum*x_step)


data_arr, label_arr = ad.load_dataset("horseColicTraining2.txt")
classifier, agg_classest = ada_boost_train(data_arr, label_arr, max_iter=10)

test_data_arr, test_label_arr = ad.load_dataset("horseColicTest2.txt")
predict = ada_classify_nosign(test_data_arr,classifier)
plot_ROC(predict.T, test_label_arr)  # 测试的ROC曲线
plot_ROC(agg_classest.T, label_arr)  # 训练的ROC曲线

