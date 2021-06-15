from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas
import numpy as np

data = pandas.read_csv('Data.csv', usecols=['X1','X2'])
result = pandas.read_csv('Data.csv', usecols=['Y'])

arrX = np.array(data)
arrY = np.array(result).flatten()

model = LinearRegression()
model.fit(arrX,arrY)

test = [[5,11],[5,12],[5,50]]

print("Correlation: ", model.score(arrX,arrY))
print("Slope: ", model.coef_)
print("B0: ",model.intercept_)
print("***********************")
print(model.predict(test))


# Plot outputs
plt.scatter(arrX, np.repeat(arrY,2), color='black')
plt.plot(arrX, model.predict(arrX), color='blue', linewidth=3)
plt.show()
