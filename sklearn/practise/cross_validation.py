# 交叉验证
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def init():
    # 加载iris数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


# 基础交叉验证
def mode():
    X, y = init()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=5) # 选取5个邻近点
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)   # 本质上，knn不需要训练
    print(score)


# k折交叉验证 k=5
def mode1(k=5):
    X, y = init()
    knn = KNeighborsClassifier(n_neighbors=5) # 选取5个邻近点
    # knn.fit(X_train, y_train)  # 此时Knn不需要调用fit 其他的ML需要训练
    scores = cross_val_score(knn, X, y, cv=k, scoring="accuracy")  # 5组数据
    print(scores)
    print(scores.mean())


def mode2(scoring="accuracy"):
    X, y = init()
    # 建立测试参数集
    k_range = range(1, 31)
    k_scores = []
    # 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring=scoring)
        k_scores.append(scores.mean())

    # 可视化数据
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def mode3():
    from sklearn.model_selection import learning_curve
    digits = load_digits()
    X, y = digits.data, digits.target
    # 观察样本由小到大的学习曲线变化, 采用K折交叉验证 cv=10,
    # 选择平均方差检视模型效能 scoring='mean_squared_error',
    # 样本由小到大分成5轮检视学习曲线(10%, 25%, 50%, 75%, 100%):
    train_sizes, train_loss, test_loss = learning_curve(
        # gramma = 核函数参数的值 float
        SVC(gamma=0.001), X, y, cv=5, scoring="neg_mean_squared_error",
            train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
        )
    # 是负值，为了看着方便可以加上负号
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.plot(train_sizes, train_loss_mean, "o-", color="r", label="Training")
    plt.plot(train_sizes, test_loss_mean, "o-", color="g", label="Cross-validation")
    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


    """
        参数调优过程    
    """
def mode4():
    from sklearn.model_selection import validation_curve
    digits = load_digits()
    X, y = digits.data, digits.target
    param_range = np.logspace(-6, 1, 10)
    train_loss, test_loss = validation_curve(
        SVC(), X, y, param_name="gamma", param_range=param_range, cv=10,
        scoring="neg_mean_squared_error"
    )
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.plot(train_loss_mean, "o-", color="r", label="Training-error")
    plt.plot(test_loss_mean, "o-", color="g", label="cross_validation-error")
    plt.xlabel("Training_error")
    plt.ylabel("cross_validation_error")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    # mode()  # 基础交叉验证
    # mode1()   # k-折交叉验证
    """
        测试准确度， 偏差
        从图中可以得知，选择12~18的k值最好。高过18之后，准确率开始下降则是因为过拟合(Over fitting)的问题。
        k值 指的是n_neighbors
    """
    # mode2()  # 重复多次，求取k- 交叉验证平均值
    """
       # 测试平均方差 方差
        由图可以得知，平均方差越低越好，因此选择13
        ~18
        左右的K值会最好。
    """
    # mode2(scoring="neg_mean_squared_error")  # 重复多次，求取k-交叉验证平均值 neg_mean_squared_error
    # mode3()
    mode4()