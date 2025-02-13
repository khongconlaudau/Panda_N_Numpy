import numpy as np


def estimate_gaussian(X):
    # Using built-in functions
    # mu = np.mean(X, axis=0)
    # var = np.var(X, axis=0)
    #
    #  Vectorization
    m,n = X.shape
    mu = np.sum(X, axis=0) / m
    var = np.sum((X.T - mu.reshape(n,-1))**2, axis =1 )/ m

    return mu, var

def select_threshold(y_val, p_val):
    best_epsilon, best_f1, f1 = 0, 0, 0
    step = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step):
        prediction = (p_val < epsilon)
        tp = np.sum((y_val == 1) & (prediction == 1))
        fp = np.sum((y_val == 0) & (prediction == 1))
        fn = np.sum((y_val == 1) & (prediction == 0))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        f1 = 2 * prec * rec / (prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1

