import numpy as np
import numbers
# import pickle
from zodbpickle import pickle
from scipy.misc import imsave


def get_reshaped_image_size(vec_image_size):
    """
    Given an image, reshaped to vector, size, checks if it is valid
    and returns the initial image size
    :param vec_image_size: the size of a vector, which is a reshaped image
    :return: the size of the initial image
    """
    if not isinstance(vec_image_size, numbers.Number):
        raise ValueError("The parameter is not a number")
    if int(vec_image_size) % 3 != 0:
        raise ValueError("The image-array has invalid shape")
    image_shape = np.sqrt(vec_image_size / 3)
    if image_shape != int(image_shape):
        raise ValueError("The image-array has invalid shape")
    return int(image_shape)


def normalize_data(images):
    """
    Normalizes input images, truncating the values exceeding the 3-sigma region
    for each channel of the image
    :param images: a numpy array of size (NxD). It contains N images as it's rows
    reshaped to vectors, which can be reshaped back to (dxdx3) arrays.
    :return: a numpy array of size (NxD) containing the normalized images.
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("The parameter 'images' must be a numpy.ndarray")
    N, D = images.shape
    d = get_reshaped_image_size(D)
    images = images.reshape((N, d, d, 3))
    for channel in range(3):
        images_channel = images[:, :, :, channel]
        mean = np.mean(images_channel)
        images_channel -= mean
        std = np.sqrt((np.sum(images_channel ** 2) / images_channel.size))
        images_channel[images_channel > 3 * std] = 3 * std
        images_channel[images_channel < - 3 * std] = - 3 * std
        images_channel *= 0.4 / (3 * std)
        images_channel += 0.5
        images[:, :, :, channel] = images_channel
    return images.reshape((N, D))


def sample_patches_raw(images, num_patches=10000, patch_size=8):
    """
    Generates unnormalized sample patches from provided images
    :param images: a numpy array of size (M, S). It contains M images in it's
    rows, reshaped to vectors, which can be reshaped back to (sxsx3) arrays.
    :param num_patches: number N of patches to generate
    :param patch_size: the length of one side of the (square) patch.
    :return: a numpy array of size (N, D), where N = num_patches,
    D = 3 * patch_size**2. It contains patches reshaped to vectors in it's rows.
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("The parameter 'images' must be a numpy.ndarray")
    if num_patches <= 0 or patch_size <= 0:
        raise ValueError("Invalid values for 'patch_size' or 'num_patches'")
    M, S = images.shape
    D = 3 * patch_size**2
    s = get_reshaped_image_size(S)
    if s < patch_size:
        raise ValueError("Patch size is bigger then the image size")
    image_indices = np.random.randint(0, M, num_patches)
    patch_corners = [np.random.randint(0, s - patch_size + 1, num_patches),
                     np.random.randint(0, s - patch_size + 1, num_patches)]
    chosen_images = images[image_indices, :].reshape((num_patches, s, s, 3))
    patches = np.zeros((num_patches, D)).reshape((num_patches, patch_size, patch_size, 3))
    for i in range(patch_size):
        for j in range(patch_size):
            for channel in range(3):
                channel_list = [channel] * num_patches
                patches[:, i, j, channel] = chosen_images[range(num_patches), patch_corners[0],
                                                    patch_corners[1], channel_list]
            patch_corners[0] = list(map(lambda x: x + 1, patch_corners[0]))
        patch_corners[0] = list(map(lambda x: x - patch_size, patch_corners[0]))
        patch_corners[1] = list(map(lambda x: x + 1, patch_corners[1]))
    return patches.reshape(num_patches, D)

def sample_patches(images, num_patches=10000, patch_size=8):
    """
    Calls the function normalize_data on what sample_patches_raw returns.
    Thus, generates normalized patches from given images.
    :param images: a numpy array of size (M, S). It contains M images in it's
    rows, reshaped to vectors, which can be reshaped back to (sxsx3) arrays.
    :param num_patches: number N of patches to generate
    :param patch_size: the length of one side of the (square) patch.
    :return: a numpy array of size (N, D), where N = num_patches,
    D = 3 * patch_size**2. It contains patches reshaped to vectors in it's rows.
    """
    patches = sample_patches_raw(images, num_patches, patch_size)
    return normalize_data(patches)

if __name__ == '__main__':
    images_dict = pickle.load(open('../data/unlabeled.pk', 'rb'))
    images = images_dict['X']
    n_patches = 10
    patch_size = 10
    patches = sample_patches(images, n_patches, patch_size)
    for i in range(n_patches):
        patch = patches[i, :].reshape(patch_size, patch_size, 3)
        imsave('patches/patch_'+str(i)+'.png', patch)