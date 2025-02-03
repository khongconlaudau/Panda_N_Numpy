import copy
import numpy as np
from sigmoid import sigmoid
from Cost_Function import compute_cost_logistic



# calc the derivative of w_j and b_j
def compute_gradient_logistic(x, y, w, b):
    m,n = x.shape # where m = # of examples, n = # of features
    dj_dw = np.zeros(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * x[i][j]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

#  calc the derivative of w_j and b_j (More Efficient)
def compute_gradient_logistic_2(x, y, w, b):
    m = x.shape[0]
    z = np.dot(x,w) + b
    err = sigmoid(z) - y
    dj_dw = np.dot(x.T, err) / m
    dj_db = np.sum(err) / m

    return dj_dw, dj_db

# implement gradient_descent
def gradient_logistic(x, y, w_in, b_in, alpha, num_iters):
    j_history = [] # store cost history
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(compute_cost_logistic(x,y,w,b))
        if i % 1000 == 0:
            print(f"Iteration: {i}, Cost: {j_history[-1]}")
    return w, b

if __name__ == '__main__':
    # Data set
    x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # tmp w and b
    w_tmp = np.zeros_like(x_train[0])
    b_tmp = 0.
    alph = 0.1
    iters = 10000

    np.set_printoptions(precision=2)
    # using gradient_logistic to find proper w and b
    w_out, b_out = gradient_logistic(x_train, y_train, w_tmp, b_tmp, alph, iters)
    print(f"\nupdated parameters: w:{w_out}, b:{b_out}")