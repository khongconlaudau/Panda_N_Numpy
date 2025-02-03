import numpy as np
from Sigmoid import sigmoid



# Computes cost over all examples with Linear Regression
def compute_cost_linear_reg(x, y, w, b, lambda_):
    m, n = x.shape # number of examples and features
    cost = 0.

    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost += (f_wb_i - y[i])**2

    cost = cost / (2*m)

    reg_cost = 0.

    for j in range(n):
        reg_cost += (w[j] **2)
    reg_cost = (lambda_ / (2*m)) * reg_cost

    return cost+reg_cost

# More efficient way to cal cost function of linear reg with Vectorization
def compute_cost_linear_reg_2(x, y, w, b, lambda_):
    m = x.shape[0]

    cost = np.sum(((np.dot(x,w)+b) - y)**2 )  / (2*m)
    reg_cost = (lambda_/(2*m)) * np.sum(w**2)
    return cost+reg_cost

# Calculate cost function of logistic regression using vectorization
def compute_cost_logistic_reg(x, y, w, b, lambda_):
    m = x.shape[0]
    z = np.dot(x,w) + b
    f_wb = sigmoid(z)

    cost = (-1/m) * np.sum(y*np.log(f_wb) + (1-y) * np.log(1-f_wb))

    reg_cost = (lambda_/(2*m)) * np.sum(w**2)

    return cost+reg_cost


# Computes the gradient for linear regression using Vectorization
def compute_gradient_linear_reg(x, y, w, b, lambda_):

    m = x.shape[0] # number of examples in the training set
    err = np.dot(x, w) + b - y
    dj_dw = (1/m) * np.dot(x.T, err) + (lambda_/m) * w
    dj_db = (1/m) *  np.sum(err)

    return dj_dw, dj_db

# Computes the gradient for Logistic Reg using Vectorization
def compute_gradient_logistic_reg(x, y, w, b, lambda_):

    m = x.shape[0]
    z = np.dot(x,w) + b
    err = sigmoid(z) - y
    dj_dw = (1/m) * np.dot(x.T, err) + (lambda_/m) * w
    dj_db = (1/m) * np.sum(err)

    return dj_dw, dj_db

if __name__ == '__main__':
    # Linear Cost using loop
    np.random.seed(1)
    X_tmp = np.random.rand(5, 6)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1, ) - 0.5
    b_tmp = 0.5
    lambda_tmp = 0.7
    cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print("Regularized cost:", cost_tmp)

    # Linear Cost using Vectorization
    cost_tmp2 = compute_cost_linear_reg_2(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
    print("Regularized cost:", cost_tmp2)


    # Logistic Cost using Vectorization
    print("Regularized cost of Logistic Reg: ",compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp))

    # Gradient Descent of Linear Reg using Vectorization
    np.random.seed(1)
    X_tmp = np.random.rand(5, 3)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1])
    b_tmp = 0.5
    lambda_tmp = 0.7
    dj_dw_tmp, dj_db_tmp = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print(f"dj_db: {dj_db_tmp}", )
    print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

    # Gradient Descent of Logistic Reg using Vectorization
    dj_dw_log, dj_db_log = compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
    print(f"dj_db of Log:\n {dj_db_log}", )
    print(f"Regularized dj_dw of Log:\n {dj_dw_log.tolist()}", )