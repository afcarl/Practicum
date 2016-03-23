import numpy as np
import math
import scipy.stats as stat


def log_multivariate_normal_density(points, mean, covariance):
    """
    A non-vectorized implementation of a function, calculating the density
    vector of the multivariate normal distribution at the given points
    """
    def matrix_dot(a, b):
        res = np.zeros((a.shape[0], b.shape[1]))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                for k in range(a.shape[1]):
                    res[i, j] += a[i, k] * b[k, j]
        return res

    d = points.shape[1]
    n = points.shape[0]
    res = np.zeros(n)
    l = np.linalg.cholesky(covariance)
    l_inv = np.linalg.inv(l)
    cov_inv = l_inv.T.dot(l_inv)
    log_det = np.sum(np.log(np.diag(l)))
    for i in range(n):
        diff = (points[i, :] - mean[None, :])
        res[i] = (-matrix_dot(matrix_dot(diff, cov_inv), diff.T)/2 -
                   d*np.log(2*math.pi)/2 - log_det)
    return res


def vec_log_multivariate_normal_density(points, mean, covariance):
    """
    Vectorized implementation of the same function
    """
    d = points.shape[1]
    diff = (points - mean[None, :])
    l = np.linalg.cholesky(covariance)
    l_inv = np.linalg.inv(l)
    cov_inv = l_inv.T.dot(l_inv)
    log_det = np.sum(np.log(np.diag(l)))
    return np.diag(-(diff.dot(cov_inv)).dot(diff.T)/2 -
                   d*np.log(2*math.pi)/2 - log_det)


def fun_log_multivariate_normal_density(points, mean, covariance):
    """
    A functional implementation of the same function
    """
    d = points.shape[1]
    n = points.shape[0]
    l = np.linalg.cholesky(covariance)
    l_inv = np.linalg.inv(l)
    cov_inv = l_inv.T.dot(l_inv)
    log_det = np.sum(np.log(np.diag(l)))
    return np.array([-(diff.dot(cov_inv)).dot(diff.T)/2 -
                     d*np.log(2*math.pi)/2 - log_det
                     for diff in [(points[i, :] - mean[None, :])
                                  for i in range(n)]]).reshape(n,)
