import numpy as np

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our
    specific required shape.
    """

    L = len(parameters) // 2
    keys = {}
    count = 0

    for l in range(L):
        # reshape weight parameter into a single vector
        new_vector = np.reshape(parameters["W" + str(l+1)], (-1, 1))
        # concat the bias vector onto the flattened weight vector
        new_vector = np.concatenate((new_vector,parameters["b" + str(l+1)]),
                axis=0)
        # add shapes of previous vector for retrieval later
        keys["W" + str(l+1)] = parameters["W" + str(l+1)].shape
        keys["b" + str(l+1)] = parameters["b" + str(l+1)].shape

        # concat new_vector into theta
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta, keys):
    """
    Return flattened vector of parameters into dictionary
    """

    L = len(keys) // 2
    start_vec = 0
    parameters = {}

    for l in range(L):
        num_weights = keys["W" + str(l+1)][0] * keys["W" + str(l+1)][1]
        parameters["W" + str(l+1)] = theta[start_vec:start_vec +
                num_weights].reshape(keys["W" + str(l+1)])
        start_vec += num_weights

        num_weights = keys["b" + str(l+1)][0] * keys["b" + str(l+1)][1]
        parameters["b" + str(l+1)] = theta[start_vec:start_vec +
                num_weights].reshape(keys["b" + str(l+1)])

    return parameters

def gradients_to_vector(grads):
    """
    Roll all our gradients dictionary into a single vector satisfying our
    specific required shape.
    """

    L = len(grads) // 3
    keys = {}
    count = 0

    for l in range(L):
        # reshape weight parameter into a single vector
        new_vector = np.reshape(grads["dW" + str(l+1)], (-1, 1))
        # concat the bias vector onto the flattened weight vector
        new_vector = np.concatenate((new_vector,grads["db" + str(l+1)]),
                axis=0)
        # add shapes of previous vector for retrieval later
        keys["dW" + str(l+1)] = grads["dW" + str(l+1)].shape
        keys["db" + str(l+1)] = grads["db" + str(l+1)].shape

        # concat new_vector into theta
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys



