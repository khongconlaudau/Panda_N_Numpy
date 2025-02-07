from Sigmoid import sigmoid
import numpy as np

# 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))(2)

def compute_cost_logistic(x, y, w, b):
    m = x.shape[0] # number of examples in the data set
    cost = 0.0

    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost

if __name__ == '__main__':
    # Data set
    x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
    y_train = np.array([0, 0, 0, 1, 1, 1])

    w_tmp = np.array([1,1])
    b_tmp = -3
    print(compute_cost_logistic(x_train, y_train, w_tmp, b_tmp))
