from coffee_utils import load_data
from tensorflow.keras.layers import Dense
from  tensorflow.keras import  Sequential
import tensorflow as tf
import numpy as np

# Data set (Made up)
x,y = load_data()

# Normalize Data
print(f"Temperature Max, Min pre normalization: {np.max(x[:,0]):0.2f}, {np.min(x[:,0]):0.2f}")
print(f"Duration Max, Min pre normalization: {np.max(x[:,1]):0.2f}, {np.min(x[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(x)
xn = norm_l(x)
print(f"Temperature Max, Min post normalization: {np.max(xn[:,0]):0.2f}, {np.min(xn[:,0]):0.2f}")
print(f"Duration Max, Min post normalization: {np.max(xn[:,1]):0.2f}, {np.min(xn[:,1]):0.2f}")


# Tile/copy our data to increase the training set size and reduce the number of training epochs.
xt = np.tile(xn,(1000,1))
yt = np.tile(y,(1000,1))


# Tensorflow Model
tf.random.set_seed(1234) #applied to achieve consistent results
model = Sequential([
    tf.keras.Input(shape=(2,)), #The tf.keras.Input(shape=(2,)), specifies the expected shape of the input(features)
    # . This allows Tensorflow to size the *weights* and bias parameters at this point
    Dense(units=8, activation='relu', name='layer1'),
    Dense(units=4, activation='relu', name='layer2'),
    Dense(units=1, activation='sigmoid', name='output'),  # Output layer
])

model.summary()
# 3 neurons with 2 features -> 3 * 2 = 6 + 3(bias)
# Layer 2: received 3 inputs, so it has 3 weights + 1 bias

# In the first layer with 3 units, we expect W to have a size of (2,3) and  𝑏
#   should have 3 elements.
# In the second layer with 1 unit, we expect W to have a size of (3,1) and  𝑏
#   should have 1 element
# ********(Input, Unit)****************
W1, b1 = model.get_layer(name='layer1').get_weights()
W2, b2 = model.get_layer(name='layer2').get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)



# ******** Epochs *******
# Defined: An epoch is one complete pass through the entire training dataset.
# In the fit statement above, the number of epochs was set to 10.
# This specifies that the entire data set should be applied during training 10 times during training


# ******** Batches *******
# Define: A batch is a subset of the training data processed at one time.
# Instead of updating the model after every sample, we update it (weights and bias) after processing a batch of samples.
# Default size: 32
model.compile(
    loss= tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)
model.fit(xt, yt, epochs=10,batch_size=100)
# w and b after fitting process
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# Prediction
X_test = np.array([
    [230, 13.9],
    [200, 17] ])   # negative example
X_test_nor = norm_l(X_test)
y_pred = (model.predict(X_test_nor) >= 0.5).astype(int)
print("Prediction: ",y_pred)