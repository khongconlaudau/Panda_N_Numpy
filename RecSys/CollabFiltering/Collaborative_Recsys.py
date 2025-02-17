import numpy as np
import  copy
import tensorflow as tf
from tensorflow import keras
def cofi_cost_fucn(X, W, b, Y, R, lambda_):
    """
        Returns the cost for the content-based filtering
        Args:
          X (ndarray (num_movies,num_features)): matrix of item features
          W (ndarray (num_users,num_features)) : matrix of user parameters
          b (ndarray (1, num_users)            : vector of user parameters
          Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
          R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
          lambda_ (float): regularization parameter
        Returns:
          J (float) : Cost
    """

    pred = np.dot(X, W.T) + b
    err = np.sum(((pred - Y ) * R) ** 2) / 2
    reg = (lambda_ / 2) * (np.sum(W**2) + np.sum(X**2))

    return err + reg


def gradient_descent(num_movies, num_users, num_features, Ynorm, R,alpha, lambda_,num_iters):
    """
       Performs gradient descent to learn X, W, and b.

       Args:
         X (ndarray): (num_movies, num_features) matrix of item features
         W (ndarray): (num_users, num_features) matrix of user parameters
         b (ndarray): (1, num_users) vector of user biases
         Ynorm (ndarray): (num_movies, num_users) matrix of user ratings
         R (ndarray): (num_movies, num_users) binary matrix where R(i, j) = 1 if movie i was rated by user j
         lambda_ (float): Regularization parameter
         alpha (float): Learning rate
         num_iters (int): Number of iterations

       Returns:
         X (ndarray): Updated item feature matrix
         W (ndarray): Updated user preference matrix
         b (ndarray): Updated bias vector
         J_history
    """

    # Set Initial Parameters (W, X), use tf.Variable to track these variables
    tf.random.set_seed(1234) # for consistent results
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
    b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')
    J_history = []
    # Instantiate an optimize
    optimizer = keras.optimizers.Adam(learning_rate=alpha)

    for i in range(num_iters):
        with tf.GradientTape() as tape:
            # Compute the cost (forward pass included in cost)
            cost_value = cofi_cost_fucn(X, W, b, Ynorm, R, lambda_)

        grads = tape.gradient(cost_value, [X,W,b])

        optimizer.apply_gradients(zip(grads, [X,W,b]))
        if i % 100 == 0:
            J_history.append(cost_value)

    return X, W, b, J_history


