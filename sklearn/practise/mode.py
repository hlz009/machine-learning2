"""
    save pickle, use to predict conviniently
"""

from sklearn.svm import SVC
from sklearn.datasets import load_digits


digits = load_digits()
X, y = digits.data, digits.target
svc = SVC(gamma=0.03)
svc.fit(X, y)  # 训练数据


import pickle
# """
#     保存训练好的模型文件
# """
# with open('svc_digits.pickle', 'wb') as f:
#     pickle.dump(svc, f)


"""
    读取训练好的模型，并开始使用
"""
# with open("svc_digits.pickle", "rb") as f:
#     svc = pickle.load(f)
#     print(svc.predict(X[0:1]))  # 注意不能写成X[0] X[0:1]shape要与原来的shape一致


"""
 使用joblib, 使用方法与pickle模块一致，
 有的说joblib内部采用了多进程，处理大数据量时，较pickle快
"""
from sklearn.externals import joblib

# with open("svc_digits_1.pickle", "wb") as f:
#     joblib.dump(svc, f)


with open("svc_digits1.pickle", "rb") as f:
    svc = joblib.load(f)
    print(svc.predict(X[0:1]))  # 注意不能写成X[0] X[0:1]shape要与原来的shape一致

