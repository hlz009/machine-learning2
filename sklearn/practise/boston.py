from sklearn import datasets
from sklearn.linear_model import LinearRegression


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)
y = model.predict(data_X[:4, :])
print(model.coef_)  # 回归系数
print(model.intercept_)  # 截距
print(model.get_params())
print(model.score(data_X, data_y))

