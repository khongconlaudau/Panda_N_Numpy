#The most important function in matplotlib is plot,
# which allows you to plot 2D data. Here is a simple example:
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)
z = np.cos(x)

# plt.plot(x,y, label="sin")
# # plt.plot(x,z, label="cos")
# # plt.xlabel('x axis label')
# # plt.ylabel('y axis label')
# # plt.legend()
# # plt.show()

plt.subplot(2, 1, 1)


plt.plot(x, y)
plt.title('Sine')

plt.subplot(2, 1, 2)

plt.plot(x, z)
plt.title('Cos')
plt.show()
print(math.comb(1000,2))
