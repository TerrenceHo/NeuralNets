import numpy as np

def Cross_Entropy_Loss(AL, Y):
    """
    Implement the cross entropy cost function.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    # Add on regularization
    return cost

def Mean_Squared_Lost(AL, Y):
    return

def Regularization(parameters, lambd, m):
    """
    Implemented Regularization to cost functions. Regularization method is l2

    Arguments:
    parameters -- dictionary of weights and biases
                        Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                        bl -- bias vector of shape (layers_dims[l], 1)
    lambd -- regularization parameter, if zero there is not regularization
    m -- number of examples

    Returns:
    reg -- regularization of weights
    """

    reg = 0.0
    L = len(parameters) // 2 # number of layers in the neural network
    if method is None:
        return 0
    elif method == 'l2':
        for l in range(L):
            reg += np.sum(np.square(parameters["W" + str(l + 1)]))
        reg *= lambd/(2.0 * m)
        return reg
    elif method == 'l1':
        pass
    return reg


