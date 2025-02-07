import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.activations import sigmoid
import matplotlib.pyplot as plt

# Data set
X_train = np.array([[1.0], [2.0]], dtype=np.float32)            #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

# defined a layer with one neuron (Linear Reg)
# ***the input to the layer must be 2-D***
linear_layer = tf.keras.layers.Dense(units=1, activation='linear',)
a1 = linear_layer(X_train[0].reshape(-1, 1))

# These weights are randomly initialized to small numbers and
# the bias defaults to being initialized to zero.
w, b= linear_layer.get_weights()

# set weights
set_w = np.array([[200]])
set_b = np.array([100])
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

# Compare linear equation to the layer output
p1 = linear_layer(X_train[0].reshape(-1, 1))
print(p1)

p_lin = np.dot(set_w,X_train[0].reshape(-1, 1)) + set_b
print(p_lin)


# Neuron with Sigmoid activation
X2_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y2_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

# pos = Y2_train == 1
# neg = Y2_train == 0
#
# _,ax = plt.subplots(1,1,figsize=(5,5))
# ax.scatter(X2_train[pos], Y2_train[pos], marker='x', s=80, c='red', label='y=1')
# ax.scatter(X2_train[neg], Y2_train[neg], marker='o', s=100, c='blue', label='y=0')
# ax.set_xlabel('Tumor Size')
# ax.set_ylabel('Prob')
# ax.legend()
# plt.show()


# Tensorflow is most often used to create multi-layer models.
# The Sequential model is a convenient means of constructing these models
model = Sequential([
    tf.keras.layers.Dense(units=1, input_dim=1, activation='sigmoid', name='L1'),
])

model.summary()
# get layer from model
logistic_layer = model.get_layer('L1')
w2,b2 = logistic_layer.get_weights()
print(w2,b2)
print(w2.shape,b2.shape )

# set way and bias for log_layer
set_w2 = np.array([[2]])
set_b2 = np.array([-4.5])
logistic_layer.set_weights([set_w2, set_b2])
print(logistic_layer.get_weights())

# Compare sigmoid equation to the log_layer output
log_layer_pred = model.predict(X2_train[0].reshape(-1, 1))
print(f"log_layer_pred : {log_layer_pred}")
log_reg_pred = sigmoid(np.dot(set_w2, X2_train[0].reshape(-1, 1))+set_b2 )
print("log_reg_pred :", log_reg_pred)