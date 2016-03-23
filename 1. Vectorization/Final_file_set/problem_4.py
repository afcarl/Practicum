import numpy as np


def maximum_in_front_of_zero(vec):
    """
    A non-vectorized implementation of a function, that finds a maximum
    of the elements of the given vector vec, that have a 0 behind them.
    """
    in_front_of_zero = False
    max_elem = -np.inf
    for i in vec:
        if in_front_of_zero:
            if i > max_elem:
                max_elem = i
        in_front_of_zero = (i == 0)
    return max_elem


def vec_maximum_in_front_of_zero(vec):
    """
    A vectorized implementation of the same function
    """
    return np.max((np.roll(vec, -1))[vec[:-1] == 0])


def fun_maximum_in_front_of_zero(vec):
    """
    A functional implementation of the same function
    """
    return max([elem for elem, prev_elem in zip(vec[1:], vec[:-1])
                if prev_elem == 0])
