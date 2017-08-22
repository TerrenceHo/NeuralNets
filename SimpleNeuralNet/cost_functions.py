import numpy as np

def Cross_Entropy_Loss(AL, Y, reg_func, parameters, derivative=False):
    """
    Implement the cross entropy cost function.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    derivative -- bool to decide if function should calculate derivative or not

    Returns:
    cost -- cross-entropy cost if derivative == True
            OR dAL, derivative of the cost function
    """
    
    print("Y = ", Y)
    print("AL = ", AL)
    if derivative:
        dAL =  -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        return dAL
    else:
        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        # Add regularization
        cost += reg_func(parameters)/m
        return cost

def Mean_Squared_Loss(AL, Y):
    return

def L2_Regularization(reg_lambd):
    """
    Implemented Regularization to cost functions. Regularization method is l2

    Arguments:
    reg_lambd -- regularization parameter, if zero there is not regularization

    Returns:
    reg_func -- function that can calculate either the l2 Regularization, or its
        derivative during back propagation
            Arguments:
                parameters -- dictionary of weights and biases
                                    Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                                    bl -- bias vector of shape (layers_dims[l], 1)
                           -- If derivative == True, then parameters is simply one matrix of
                              weights, if only Wl.
                m -- number of examples
    """


    def reg_func(parameters, derivative=False):
        if derivative:
            return parameters * (reg_lambd)
        else:
            L = len(parameters) // 2 # number of layers in the neural network
            reg = 0.0
            for l in range(L):
                reg += np.sum(np.square(parameters["W" + str(l + 1)]))
            reg *= reg_lambd/(2.0)
            return reg

    return reg_func


