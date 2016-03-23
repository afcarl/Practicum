import numpy as np
import pickle
from scipy.misc import imsave

from sample_patches import get_reshaped_image_size, sample_patches


def display_layer(X, filename="../images/layer.png"):
    """
    Produces an image, composed of the given N images, patches or neural network weights,
    stored in the array X. Saves it with the given filename.
    :param X: numpy array of size (NxD) — N images, patches or neural network weights
    :param filename: a string, the name of the produced file
    :return: None
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("'X' must be a numpy array")
    N, D = X.shape
    d = get_reshaped_image_size(D)

    if N == 1:
        return X.reshape(d, d, 3)
    divizors = [n for n in range(1, N) if N % n == 0]
    im_sizes = divizors[int(len(divizors) / 2)], int(N / divizors[int(len(divizors) / 2)])
    for i in range(im_sizes[0]):
        # img_row = np.hstack((img_row, np.zeros((d, 1, 3))))
        img_row = np.hstack((np.zeros((d, 1, 3)), np.array(X[i * im_sizes[0], :].reshape(d, d, 3))))
        img_row = np.hstack((img_row, np.zeros((d, 1, 3))))
        for j in range(1, im_sizes[1]):
            img_row = np.hstack((img_row, X[i * im_sizes[1] + j, :].reshape(d, d, 3)))
            img_row = np.hstack((img_row, np.zeros((d, 1, 3))))
        if i == 0:
            img = img_row
        else:
            img = np.vstack((img, img_row))
        img = np.vstack((img, np.zeros((1, img.shape[1], 3))))
    img = np.vstack((np.zeros((1, img.shape[1], 3)), img))
    imsave(filename, img)
    return img

if __name__ == '__main__':
    images_dict = pickle.load(open('../data/train.pk', 'rb'))
    images = images_dict['X']
    n_patches = 1000
    patch_size = 10
    patches = sample_patches(images, n_patches, patch_size)
    display_layer(patches)