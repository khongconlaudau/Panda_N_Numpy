import numpy as np
import copy, math

# data is stored in numpy array/matrix
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083 #bias
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

#  single predict using linear regression
def single_predict(x, w, b):
    p = np.dot(x, w) + b
    return p # estimated cost

# get a row from our training data
x_vect = x_train[0, :]
f_wb = single_predict(x_vect, w_init, b_init)
print(f"Cost: {f_wb}")

# compute cost for multiple features
def compute_cost(x, y, w, b):
    m = x.shape[0] # number of examples
    cost = 0.0

    for i in range(m):
        f_wb_i = single_predict(x[i], w, b)
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

J_wb = compute_cost(x_train, y_train,w_init , b_init)
print(f'Cost at optimal w : {J_wb}')
# Computes the gradient for linear regression
def compute_gradient(x,y, w, b):

    m, n = x.shape # number of examples and n features
    dj_dw = np.zeros(n,)
    dj_db = 0.

    for i in range(m):
        err =  single_predict(x[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i][j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

#Compute and display gradient
tmp_dj_dw, tmp_dj_db= compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


# Crazy
def compute_gradient_vectorized(x, y, w, b):
    m = x.shape[0]  # Number of training examples
    err = np.dot(x, w) + b - y  # Compute all errors at once
    dj_dw = (1 / m) * np.dot(x.T, err)  # Compute gradient for w
    dj_db = (1 / m) * np.sum(err)  # Compute gradient for b

    return dj_dw, dj_db

# Compute and display vectorized gradient
tmp_dj_dw, tmp_dj_db = compute_gradient_vectorized(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b (vectorized): {tmp_dj_db}')
print(f'dj_dw at initial w,b (vectorized): \n{tmp_dj_dw}')

def gradient_descent(x,y,w_in,b_in,alpha,num_iters):
    j_history = [] # store cost J at each iteration
    w = copy.deepcopy(w_in) #avoid modifying global w within function
    b  = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient_vectorized(x, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(compute_cost(x, y, w, b))
            if i % 100 == 0:
                print(f"Iteration: {i}, Cost: {j_history[-1]}")

    return w, b, j_history

init_w = np.zeros_like(w_init)
init_b = 0.
# Settings
iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_history = gradient_descent(x_train, y_train, init_w, init_b, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

