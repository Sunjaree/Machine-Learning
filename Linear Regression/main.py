from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([5, 15, 25, 35, 45, 55]).reshape(-1, 1)
y = np.array([5, 15, 25, 35, 45, 55])
test_array = np.array([1, 2 ,3,86,38]).reshape(-1,1)

model = LinearRegression()
model.fit(x,y)

print(model.score(x,y))
print("y intercept: ", model.intercept_)
print("\nslope: ", model.coef_)

print("**********************")

y_pred = model.predict(test_array)
print(y_pred)


