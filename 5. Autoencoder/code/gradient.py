import numpy as np


def compute_gradient(J, theta):
    """
    Computes a finite-difference approximation of the gradient of the given
    function at given point.
    :param J: function
    :param theta: a numpy array of size (d, 1)
    :return: the finite-difference approximation of the gradient of J at point theta
    """
    if not hasattr(J, '__call__'):
        raise TypeError("'J' must be callable")
    if not isinstance(theta, np.ndarray):
        raise TypeError("'theta' must be a numpy array")
    if len(theta.shape) > 1:
        if theta.shape[1] != 1:
            raise ValueError("'theta' must be a vector")

    diff = 1e-6
    diff_vec = np.zeros(theta.shape)
    grad = np.zeros(theta.shape)
    for i in range(theta.size):
        diff_vec[i] = diff
        grad[i] = (J(theta + diff_vec) - J(theta)) / diff
        diff_vec[i] = 0
    return grad

def check_gradient():
    """
    A function to make sure, that compute_gradient works correctly
    :return: true if compute_gradient works correctly and
    false otherwise
    """
    x = np.random.rand(3,1)
    return np.allclose(np.cos(x), compute_gradient(lambda point: np.sum(np.sin(point)), x), atol=1e-2)

if __name__ == '__main__':
    print(check_gradient())