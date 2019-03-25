import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier    # 使用KNN近邻算法

"""
    data, target, target_names = load_data(module_path, 'iris.csv')
"""
# 第一步：读取数据（在这之前，要收集数据并分析数据）
iris = datasets.load_iris()  # 加载数据
iris_X = iris.data  # 数据集
iris_y = iris.target  # labels 0 1 2
# print(iris_X)
# print(iris_y)

#
# 把数据集分成两个部分 训练集和测试集
# test_size 数据集中抽出30%比例作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3
)

# 此时的y_train 输出顺序也被打乱，利于学习模型
# print(y_train)

# 第二步：建立模型，训练，测试
# sklearn 提供了很多算法模型，直接调用即可
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # 训练模型
y = knn.predict(X_test)  # 预测对应的数据集，学习阶段为测试集
print(y_test == y)  #查看预测值与实际值的差别
print(np.sum(y_test != y))  # 统计有出入数据的个数
