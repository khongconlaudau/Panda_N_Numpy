import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

if __name__ == '__main__':
    # Generate an array range between -10 and 10
    z_tmp = np.arange(-10,11)

    # Using formula sigmoid
    y = sigmoid(z_tmp)

    # Plot z vs sigmoid(z)
    _,ax = plt.subplots(1,1,figsize=(5,3))
    ax.plot(z_tmp, y, c="b")
    ax.set_title("Sigmoid function")
    ax.set_ylabel('sigmoid(z)')
    ax.set_xlabel('z')
    plt.show()
