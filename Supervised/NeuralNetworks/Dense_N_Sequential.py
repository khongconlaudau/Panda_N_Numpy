import numpy as np
from Supervised.LogisticRegression.Sigmoid import sigmoid
# Implemented Dense function
a = np.array([200,17])
a1 = np.array([[200,17]])
print(a.shape)
print(a1.shape)
def my_dense(x_in, W, B):
    units = x_in.shape[1]  #
    a_out = np.zeros(units)

    for i in range(units):
        z = np.dot(x_in, W[:,i]) + B[i]
        a_out[i] = sigmoid(z)
    return a_out

# Dense w Vectorization
def my_dense_v(x_in,W,B):
    Z = np.matmul(x_in,W) + B
    a_out = sigmoid(Z)
    return a_out


def my_sequential(x,W1, b1, W2, b2 ):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2