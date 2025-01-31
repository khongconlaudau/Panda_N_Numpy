import math, copy
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):

    # number of data
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1/ (2 * m) * cost

    return total_cost
"""
 Computes the gradient for linear regression
 Args:
   x : features
   y : target values
   w,b (scalar)    : model parameters

"""

# Load data set
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_gradient(x,y, w, b):
    # Number of training example
    m = x.shape[0]
    d_dw = 0 # derivative partial of w
    d_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        d_dw += dj_dw_i
        d_db += dj_db_i
    d_dw = d_dw / m
    d_db = d_db / m

    return d_dw,d_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    # An array to store cost J at each iteration primarily for graphing later
    J_history = []

    w = w_in
    b = b_in

    for i in range(num_iters):
        d_dw, d_db = compute_gradient(x, y, w, b)

        w = w - alpha * d_dw
        b = b - alpha * d_db

        if i < 10000:
            J_history.append(compute_cost(x,y,w,b))
    return w, b, J_history