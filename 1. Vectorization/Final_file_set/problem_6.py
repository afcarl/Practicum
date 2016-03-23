import numpy as np


def run_length_encoding(vec):
    """
    A non-vectorized implementation of a function,
    performing run-length encoding of the vector vec
    """
    elem_lst, num_lst = [], []
    prev_elem = vec[0]
    num = 0
    for elem in vec:
        if elem == prev_elem:
            num += 1
        else:
            elem_lst += [prev_elem]
            num_lst += [num]
            prev_elem = elem
            num = 1
    elem_lst += [prev_elem]
    num_lst += [num]
    return np.array(elem_lst), np.array(num_lst)


def vec_run_length_encoding(vector):
    """
    A vectorized implementation of the same function
    """
    vec = vector + 1
    elems = vec[np.roll(vec, 1) != vec]
    if (elems.size == 0):
        return np.array([vec[0] - 1]), np.array([vec.size])
    reverse_cumsum = (np.cumsum(vec[::-1]))[::-1]
    elem_reverse_cumsum = reverse_cumsum[np.roll(vec, 1) != vec]
    denom = np.copy(elems)
    denom[elems == 0] = 1
    elem_count = (elem_reverse_cumsum -
                  np.hstack((np.roll(elem_reverse_cumsum, -1)[:-1], 0))) / denom
    return elems-1, elem_count.astype(int)


def fun_run_length_encoding(vec):
    """
    A functional implementation of the same function
    """
    def fun(i, elem):
        if (i >= len(vec)) or (vec[i] != elem):
            return 0
        return 1 + fun(i + 1, elem)

    elems = np.array([vec[0]] + [elem for elem, prev in
                                 list(zip(vec[1:], vec[:-1])) if elem != prev])
    indices = [0] + [i for i in range(1, len(vec))
                     if vec[i] != vec[i-1]]
    return (elems,
            np.array([fun(i, elem) for i, elem in list(zip(indices, elems))]))