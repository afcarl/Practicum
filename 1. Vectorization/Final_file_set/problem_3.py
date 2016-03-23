import numpy as np
from functools import reduce


def equal_multiset(vec_1, vec_2):
    """
    A non-vectorized implementation of a function, that
    returns True iff vec_1 and vec_2 represent the same multiset.
    """
    lst_1, lst_2 = sorted(list(vec_1)), sorted(list(vec_2))
    if len(lst_1) != len(lst_2):
        return False
    for i, j in list(zip(lst_1, lst_2)):
        if i != j:
            return False
    return True


def vec_equal_multiset(vec_1, vec_2):
    """
    A vectorized implementation of the same function
    """
    if vec_1.shape != vec_2.shape:
        return False
    return (np.sort(vec_1) == np.sort(vec_2)).all()


def fun_equal_multiset(vec_1, vec_2):
    """
    A functional implementation of the same function
    """
    return (len(vec_1) == len(vec_2)) & \
           (reduce(lambda x, y: x & (y[0] == y[1]),
                   list(zip(sorted(list(vec_1)), sorted(list(vec_2)))), True))
