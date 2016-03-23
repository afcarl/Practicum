import numpy as np


def get_vector(matr, vec_1, vec_2):
    """
    Non-vectorized implementation of the function, calculating
    a vector of a matrix' elements, given two vectors of indices.
    """
    res = []
    for it in range(len(vec_1)):
        res.append(matr[vec_1[it], vec_2[it]])
    return np.array(res)


def vec_get_vector(matr, vec_1, vec_2):
    """
    A vectorized implementation of the same function
    """
    return matr[vec_1, vec_2]


def fun_get_vector(matr, vec_1, vec_2):
    """
    A functional implementation of the same function
    """
    return np.array([matr[a, b] for a, b in list(zip(vec_1, vec_2))])
