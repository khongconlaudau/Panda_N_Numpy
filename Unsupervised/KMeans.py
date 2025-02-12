import numpy as np
from numpy import linalg as LA
from scipy.cluster.hierarchy import centroid
from tensorflow import newaxis


def find_closet_centroids(X, centroids):
    """
    :param X:(ndarray): (m, n) Input values
    :param centroids:(ndarray): (K, n) centroids
    :return:  idx (array_like): (m,) closest centroids
    """
    # idx = np.zeros(X.shape[0])
    # Without Vectorization
    # for i in range(X.shape[0]):
    #     distance = []
    #     for j in range(len(centroids)):
    #         distance.append(LA.norm(X[i] - centroids[j]))
    #     idx[i] = np.argmin(distance)

    # With vectorization

    distance = LA.norm(X[:,np.newaxis,:] - centroids, axis=2)
    idx = np.argmin(distance, axis=1)
    return idx

def compute_centroids(X, idx, k):
    """
    :param X:    (m, n) Data points
    :param idx:(m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
    :param k: number of centroids
    :return: centroids (ndarray): (K, n) New centroids computed
    """
    # Without Vectorization
    # m, n = X.shape
    # centroids = np.zeros((k, n))
    # c_k = np.zeros(k)
    # for i in range(m):
    #     centroids[idx[i]] += X[i]
    #     c_k[idx[i]] += 1
    # return centroids / c_k[:,np.newaxis]

    # with Vectorization
    centroids = np.zeros((k, X.shape[1]))
    np.add.at(centroids, idx,X)
    counts = np.bincount(idx,minlength=k).reshape(k,1)

    counts[counts == 0] = 1

    return centroids / counts

def kMeans_init_centroids(X, k):
    """
    :param X:Data points
    :param k:number of centroids
    :return: initialized centroids
    """

    randidx = np.random.permutation(X.shape[0])

    return X[randidx[:k]]

def run_KMeans(X, initial_centroids, max_iter=10):
    m, n = X.shape
    idx = np.zeros(m)
    centroids = initial_centroids
    k = initial_centroids.shape[0]
    for i in range(max_iter):
        idx = find_closet_centroids(X, centroids)

        centroids = compute_centroids(X, idx, k)

    return idx, centroids

