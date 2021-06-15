from sklearn.linear_model import LinearRegression
import numpy as np

x = [[1,2],[3,4],[5,6]]
y = [1,2,3]

x = np.array(x)
y = np.array(y)

test = np.array([[7,8],[9,10],[11,12]])

model = LinearRegression()
model.fit(x,y)

print(model.score(x,y))
print("\n")
print(model.intercept_)
print("\n")
print(model.coef_)
print("**************")

print(model.predict(test))