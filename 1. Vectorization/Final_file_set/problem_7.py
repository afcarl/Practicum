import numpy as np
import scipy.spatial.distance as sp

def find_pairwise_distances(x, y):
    """
    Finds a matrix of pairwise euclidean distances between columns of two
    given matrices x and y
    """
    res = np.zeros((x.shape[0], y.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, y.shape[0]):
            for k in range(0, x.shape[1]):
                res[i, j] += (x[i, k] - y[j, k])**2
            res[i, j] = np.sqrt(res[i, j])
    return res


def vec_find_pairwise_distances(x, y):
    """
    A vectorized implementation of the same function
    """
    return np.linalg.norm((x.T)[:, :, None] - (y.T)[:, None, :], axis=0)


def fun_find_pairwise_distances(x, y):
    """
    A functional implementation of the same function
    """
    return np.array([[np.linalg.norm(x_elem - y_elem)
                      for y_elem in np.vsplit(y, y.shape[0])]
                     for x_elem in np.vsplit(x, x.shape[0])])

# if __name__ == '__main__':
#     X = np.array([[1, 3], [2, 4]])
#     Y = np.array([[7, 3], [9, 4]])
#     Z = np.array([[3], [4]])
#
#     print(sp.cdist(X, Y))
#     print(find_pairwise_distances(X, Y))
#     print(vec_find_pairwise_distances(X, Y))
#     print(fun_find_pairwise_distances(X, Y))
