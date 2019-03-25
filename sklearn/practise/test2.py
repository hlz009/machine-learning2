from sklearn import preprocessing
import numpy as np

from sklearn.model_selection import train_test_split

# 生成样本模块
from sklearn.datasets.samples_generator import make_classification
# support vector machine 加了个惩罚因子
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 生成具有2种属性的300笔数据
x,y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                          n_informative=2, random_state=22, n_clusters_per_class=1,
                          scale=100)

# plt.scatter(x[:,0], x[:,1], c=y)
# plt.show()
# 数据标准化
# print(x)
x = preprocessing.scale(x)
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf = SVC()
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))




