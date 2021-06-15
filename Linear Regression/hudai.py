import matplotlib.pyplot as plt
import numpy as np

arrX = [[1,2],[3,4],[5,6]]
arrY = [7,8,9]


print(np.repeat(arrY,2))

plt.scatter(arrX, arrY, color = 'black')


plt.scatter(arrX, np.repeat(arrY, 2))
plt.show()
