import numpy as np
import matplotlib.pyplot as plt

# Data
x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
# g(x0 + x1 -3)
x0 = np.arange(0,6)
x1 = 3 - x0
# Features
f1 = x_train[:,0]
f2 = x_train[:,1]

positive = y_train == 1
negative = y_train == 0


plt.scatter(f1[positive], f2[positive],marker='o', c='red', s=100)  # Color by class (y_train)
plt.scatter(f1[negative], f2[negative],marker='x', c='blue', s=100)  # Color by class (y_train)
# Boundary
plt.plot(x0,x1,c='blue',alpha=0.5)
plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.title("Scatter Plot of Training Data")

# Plot the decision boundary
plt.show()