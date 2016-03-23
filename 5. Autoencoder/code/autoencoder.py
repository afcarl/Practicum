import numpy as np
import numbers
from gradient import compute_gradient
from sample_patches import get_reshaped_image_size, normalize_data


def check_hidden_size(hidden_size):
    """
    Checks if the given parameter hidden_size is a symmetric
    numpy vector
    :param hidden_size: a symmetric numpy vector, the sizes of the hidden layers
    :return: None
    """
    if not isinstance(hidden_size, np.ndarray):
        raise ValueError("'hidden_size' must be a numpy array")
    if len(hidden_size.shape) > 1:
        if hidden_size.shape[1] != 1:
            raise ValueError("'hidden_size' must be a vector")
    if np.sum(np.abs(np.fliplr(hidden_size[None, :]) - hidden_size)) != 0:
        raise ValueError("'hidden_size' must be symmetric")


def check_visible_size(visible_size):
    """
    Checks if the given parameter visible_size is a positive integer
    :param visible_size: a positive integer
    :return: None
    """
    if not isinstance(visible_size, numbers.Integral):
        raise TypeError("'visible_size' must be an integer")
    if visible_size <= 0:
        raise ValueError("'visible_size' must be positive")


def check_float(number, name="parameter"):
    """
    Checks if the given parameter is a floating-point number
    :param number: a floating-point number
    :param name: a name to be displayed in the error-message
    :return: None
    """
    if not isinstance(number, numbers.Real):
        raise TypeError("'" + name + "' must be a float")


def check_data(data):
    """
    Checks if the given parameter data is a numpy array
    :param data: a numpy array
    :return: None
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' must be a numpy array")


def reshape_to_vector(weight_lst):
    """
    Returns a vector, containing the vector containing all of the
    elements of the given list
    :param weight_lst: list of matrices and vectors of the weights
    (of a neural net)
    :return: a vector
    """
    if not isinstance(weight_lst, list):
        raise TypeError("'weight_lst' must be a list")

    if not isinstance(weight_lst[0], np.ndarray):
        raise TypeError('The weight list must contain numpy arrays only')
    weight_vec = weight_lst[0]
    weight_vec = weight_vec.reshape(-1)
    for elem in weight_lst[1:]:
        if not isinstance(elem, np.ndarray):
            raise TypeError('The weight list must contain numpy arrays only')
        weight_vec = np.concatenate((weight_vec, elem.reshape(-1)))
    return weight_vec


def check_vector(vec, name='parameter'):
    """
    Checks if the given parameter is a numpy vector
    :param weight_vec: numpy vector
    :param name: name to be shown in the error message
    :return: None
    """
    if not isinstance(vec, np.ndarray):
        raise TypeError("'" + name + "' must be a numpy array")
    if len(vec.shape) > 1:
        raise ValueError("'" + name + "' must be a vector")


def get_weights_from_vector(weight_vec, hidden_size, visible_size):
    """
    Recovers the weight-matrices list from the vector, produced
    by reshape_to_vector function
    :param weight_vec: a numpy vector of weights
    :param hidden_size: a symmetric numpy vector, the sizes of the hidden layers
    :param visible_size: int, the number of neurons in the input and output layers
    :return: a list of numpy arrays, containing weights of the autoencoder
    """
    check_hidden_size(hidden_size)
    hidden_size = hidden_size.reshape(-1)
    check_visible_size(visible_size)
    check_vector(weight_vec)
    low = 0
    high = visible_size * hidden_size[0]
    weight_lst = [weight_vec[low : high].reshape((visible_size, hidden_size[0]))]
    for i in range(hidden_size.size - 1):
        low = high
        high += hidden_size[i]
        weight_lst.append(weight_vec[low : high][:, None])
        low = high
        high += hidden_size[i] * hidden_size[i + 1]
        weight_lst.append(weight_vec[low : high].reshape((hidden_size[i], hidden_size[i + 1])))
    low = high
    high += hidden_size[-1]
    weight_lst.append(weight_vec[low : high][:, None])
    low = high
    high += hidden_size[-1] * visible_size
    weight_lst.append(weight_vec[low:high].reshape((hidden_size[-1], visible_size)))
    low = high
    weight_lst.append(weight_vec[low:][:, None])
    return weight_lst


def initialize(hidden_size, visible_size):
    """
    Randomly initializes the weights of the neural net and returns them
    reshaped and concatenated as one numpy vector
    :param hidden_size: a symmetric numpy vector, the sizes of the hidden layers
    :param visible_size: int, the number of neurons in the input and output layers
    :return: a vector, containing all the weights
    """
    check_visible_size(visible_size)
    check_hidden_size(hidden_size)
    hidden_size = hidden_size.reshape(-1)
    num_layers = hidden_size.size
    weights = []
    scale = np.sqrt(6 / (visible_size + hidden_size[0] + 1))
    weights.append(np.random.uniform(-scale, scale, (visible_size, hidden_size[0])))
    for i in range(num_layers - 1):
        weights.append(np.zeros((hidden_size[i], 1)))
        scale = np.sqrt(6 / (hidden_size[i] + hidden_size[i+1] + 1))
        weights.append(np.random.uniform(-scale, scale, (hidden_size[i], hidden_size[i+1])))
    weights.append(np.zeros((hidden_size[-1], 1)))
    scale = np.sqrt(6 / (visible_size + hidden_size[-1] + 1))
    weights.append(np.random.uniform(-scale, scale, (hidden_size[-1], visible_size)))
    weights.append(np.zeros((visible_size, 1)))
    return reshape_to_vector(weights)


def rectified_linear_unit(arr):
    """
    Apply Rectified Linear Unit function to all entries of the numpy array arr
    entries.
    :param arr: numpy array
    :return: RLU function, applied entry-wise to the parameter arr
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("'arr' must be a numpy array")
    arr[arr < 0] = 0.0
    return arr.astype(float)


def rlu_prime(arr):
    """
    Apply Rectified Linear Unit derivative function to all entries of the numpy array arr
    entries.
    :param arr: numpy array
    :return: RLU derivative function, applied entry-wise to the parameter arr
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("'arr' must be a numpy array")
    arr[arr <= 1e-15] = 0
    arr[arr > 1e-15] = 1.
    return arr.astype(float)


def sigmoid(arr):
    """
    Apply sigmoid function to all entries of the numpy array arr
    entries.
    :param arr: numpy array
    :return: sigmoid function, applied entry-wise to the parameter arr
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("'arr' must be a numpy array")
    return 1 / (1 + np.exp(-arr))


def sigmoid_prime(arr):
    """
    Apply sigmoid derivative function to all entries of the numpy array arr
    entries.
    :param arr: numpy array
    :return: sigmoid derivative function, applied entry-wise to the parameter arr
    """
    return np.exp(arr) / (1 + np.exp(arr))**2


def autoencoder_loss(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data, rlu=False):
    """
    Returns the loss of the auto-encoder on the given data set and it's gradient
    for the given weights.
    :param theta: a vector of auto-encoder weights, built as the one, returned by the
    initialize function
    :param visible_size: int, the number of neurons in the input and output layers
    :param hidden_size: a symmetric numpy vector, the sizes of the hidden layers
    :param lambda_: float, regularization parameter
    :param sparsity_param: float, sparsity parameter
    :param beta: float, regularization parameter for sparsity
    :param data: numpy array of size (NxD), containing N patches as it's rows
    :return: a tuple, loss and it's gradient at point theta
    """
    check_visible_size(visible_size)
    check_hidden_size(hidden_size)
    check_data(data)
    hidden_size = hidden_size.reshape(-1)
    check_float(beta, 'beta')
    check_float(lambda_, 'lambda_')
    check_float(sparsity_param, 'sparcity_param')
    check_vector(theta, 'theta')

    if rlu:
        f = rectified_linear_unit
        f_prime = rlu_prime
    else:
        f = sigmoid
        f_prime = sigmoid_prime

    N, D = data.shape
    d = get_reshaped_image_size(D)
    weights = get_weights_from_vector(theta, hidden_size, visible_size)
    num_layers = 2 + hidden_size.size
    layer_input = data.T
    z_lst = []
    activation_values = [data.T]
    rhos = []
    # forward pass

    loss = 0
    for weight_i in range(0, 2 * (num_layers - 1), 2):
        z_lst.append(weights[weight_i].T.dot(layer_input) + weights[weight_i + 1])
        layer_output = f(z_lst[-1])
        # print('z_lst:\n', z_lst[-1])
        # print('layer_output:\n', layer_output)
        activation_values.append(layer_output)
        # print('activation values:\n', activation_values[-1])
        layer_input = layer_output
        loss += lambda_ * np.sum(weights[weight_i]**2) / 2
        if not rlu:
            if weight_i != 2 * (num_layers - 2):
                rho_j = np.sum(layer_output, axis=1) / N
                loss += beta * np.sum(sparsity_param * np.log(sparsity_param / (rho_j + 1e-15))
                                      + (1 - sparsity_param) * np.log((1 - sparsity_param) / (1 - rho_j)) + 1e-15)
                rhos.append(rho_j[:, None])
    network_output = layer_output
    loss += np.sum((network_output - data.T)**2) / (2 * N)

    # backward pass (back propagation)
    layer = len(z_lst) - 1
    deltas = [(network_output - data.T) * f_prime(z_lst[layer]) / N]
    for weight_i in range(2 * (num_layers - 2), 0, -2):
        layer -= 1
        delta = weights[weight_i].dot(deltas[-1])
        if not rlu:
            delta += beta * (-(sparsity_param / rhos[layer]) + (1 - sparsity_param)/(1 - rhos[layer]))/N
        delta *= f_prime(z_lst[layer])
        deltas.append(delta)
        # deltas.append((weights[weight_i].dot(deltas[-1])
        #                + beta * (-(sparsity_param / rhos[layer]) + (1 - sparsity_param)/(1 - rhos[layer]))/N)
        #                * f_prime(z_lst[layer]))
    deltas.reverse()
    gradient = []
    for weight_i in range(0, len(weights), 2):
        layer = int(weight_i / 2)
        # print('deltas', deltas[layer])
        # print('activation values:\n', activation_values[layer])
        gradient.append(activation_values[layer].dot(deltas[layer].T) + lambda_ * weights[weight_i])
        gradient.append((np.sum(deltas[layer], axis=1)[:, None]))

    return loss, reshape_to_vector(gradient)


def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data, rlu=False):
    """
    Pass the given data through the neural network layers 1, ..., layer_number. The net is
    parametrized by the vector theta and it's sizes.
    :param theta: a vector of auto-encoder weights, built as the one, returned by the
    initialize function
    :param visible_size: int, the number of neurons in the input and output layers
    :param hidden_size: a symmetric numpy vector, the sizes of the hidden layers
    :param layer_number: int, the number of the layer, which we use as the output layer
    :param data: numpy array of size (NxD), containing N patches as it's rows
    :return:
    """
    check_visible_size(visible_size)
    check_hidden_size(hidden_size)
    check_data(data)
    hidden_size = hidden_size.reshape(-1)
    check_vector(theta, 'theta')
    if not isinstance(layer_number, numbers.Integral):
        raise TypeError("'layer_number' must be an integer")
    if layer_number < 0:
        raise ValueError("'layer_number' must be non-negative")
    if layer_number >= 2 + hidden_size.size:
        raise ValueError("'layer_number' is bigger then the number of layers")

    if rlu:
        f = rectified_linear_unit
    else:
        f = sigmoid

    weights = get_weights_from_vector(theta, hidden_size, visible_size)
    layer_input = data.T
    layer_output = data.T

    for weight_i in range(0, 2 * layer_number, 2):
        layer_output = f(weights[weight_i].T.dot(layer_input) + weights[weight_i + 1])
        layer_input = layer_output
    return layer_output


def autoencoder_get_filters(theta, visible_size, hidden_size, rlu=False):
    """
    Get the filters for vizualiztion of the first hidden layer of the network.
    :param theta: a vector of auto-encoder weights, built as the one, returned by the
    initialize function
    :param visible_size: int, the number of neurons in the input and output layers
    :param hidden_size: a symmetric numpy vector, the sizes of the hidden layers
    :return:
    """
    check_visible_size(visible_size)
    check_hidden_size(hidden_size)
    check_vector(theta, 'theta')
    weights = get_weights_from_vector(theta, hidden_size, visible_size)
    W = weights[0]
    patch_size = int(np.sqrt(visible_size / 3))

    filters = np.zeros((hidden_size[0], patch_size * patch_size * 3))
    print(patch_size)
    for i in range(hidden_size[0]):
        filters[i] = W[:, i]
    if rlu:
        filters[filters < 0] = 0
    # filters = filters.reshape((hidden_size[0], patch_size * patch_size * 3))
    return normalize_data(filters)


if __name__ == '__main__':
    v_size, h_size = 3, np.array([1])
    theta = initialize(h_size, v_size)
    # print(theta)
    data = np.array([[1, 2, 3], [4, 5, 6]])
    # data = np.arange(4,7).reshape(1, 3)
    beta = 0
    sparcity = 1e-2
    lambda_ = 0

    # INITIALIZATION CHECK

    # print(theta)
    # print(get_weights_from_vector(theta, h_size, v_size))

    # GRADIENT CHECK

    # grad_1 = autoencoder_loss(theta, v_size, h_size, lambda_, sparcity, beta, data, rlu=True)[1]
    # exit(1)
    # print(autoencoder_loss(theta, v_size, h_size, lambda_, sparcity, beta, data, rlu=True))
    # def J(x):
    #     return autoencoder_loss(x, v_size, h_size, lambda_, sparcity, beta, data, rlu=True)[0]
    # grad_2 = compute_gradient(J, theta)
    # print('Diff\n', get_weights_from_vector(grad_2 - grad_1, h_size, v_size))
    # print('Evaluated\n', get_weights_from_vector(grad_1, h_size, v_size))
    # print('Approx\n', get_weights_from_vector(grad_2, h_size, v_size))
    # print(theta.shape)
    # print(np.linalg.norm(grad_1 - grad_2))

    # TRANSFORM CHECK

    # print(autoencoder_transform(theta, v_size, h_size, 1, data, rlu=True))
