import numpy as np
from functools import reduce


def non_zero_prod(matr):
    """
    Non-vectorized implementation of the function, calculating the
    product of nonzero elements on the matrix' diagonal
    """
    prod = 1
    for i in range(min(matr.shape)):
        if matr[i, i] != 0:
            prod *= matr[i, i]
    return prod


def vec_non_zero_prod(matr):
    """
    Vectorized implementation of the same function
    """
    x_diag = np.diagonal(matr)
    return x_diag[x_diag != 0].prod()


def fun_non_zero_prod(matr):
    """
    Functional implementation of the same function
    """
    return reduce(lambda a, b: a*b + a*int(a*b == 0), [matr[i, i] for i in
                  range(min(matr.shape))], 1)
