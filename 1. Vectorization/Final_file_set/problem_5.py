import numpy as np
from functools import reduce


def add_channels(img, weights):
    """
    A non-vectorized implementation of a function, that
    adds up all channels of the given image with given weights
    """
    result = np.zeros(img[:, :, 0].shape)
    for i in range(len(weights)):
        weight = weights[i]
        chanel = img[:, :, i]
        for a in range(result.shape[0]):
            for b in range(result.shape[1]):
                result[a, b] += weight * chanel[a, b]
    return result[:, :]


def vec_add_channels(img, weights):
    """
    A vectorized implementation of the same function
    """
    return np.average(img, axis=2, weights=weights)


def fun_add_channels(img, weights):
    """
    A functional implementation of the same function
    """
    return (reduce(lambda a, b: a+b,
                   [weight * channel for weight, channel in list(
                    zip(weights, np.dsplit(img, img.shape[2])))]))[:, :, 0]
