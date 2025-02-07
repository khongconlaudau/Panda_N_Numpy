import numpy as np


# Implemented softmax function
def my_softmax(z):
    ez = np.exp(z)
    a_out = ez / np.sum(ez)

    return a_out


