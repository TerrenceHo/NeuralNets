import numpy as np
from scipy.special import expit

"""
Defines activation functions for connected layers in a network
sigmoid, tanh, and relu include derivatives
Softmax does not include derivatives, because it should only be used for the
last layer since it is numerically unsable
"""
# def softmax(w):
#     """Calculates the softmax function that outputs a vector of values that sum to one.
#         We take max(softmax(v)) to be the predicted label. The output of the softmax function
#         is also used to calculate the cross-entropy loss
#     """
#     logC = -np.max(w)
#     return np.exp(w + logC)/np.sum(np.exp(w + logC), axis = 0)

# def tanh(w, derivative=False):
#     """
#     Computes vector wise tanh or its derivative
#     """
#     if derivative:
#         return 1 - np.square(np.tanh(w))
#     return np.tanh(w)


def sigmoid(Z, derivative=False):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- 'Z' where we store for computing backward propagation efficiently. Also
        called cache
    derivative -- bool to determine if we are computing derivative or not 

    Returns:
    if derivative == False:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
    if derivative == True:
        dZ -- Gradient of the cost with respect to Z
    """

    if derivative:
        # s = np.where(Z > 0, 1. / (1. + np.exp(-Z)), np.exp(Z) / (np.exp(Z) + np.exp(0)))
        s = expit(Z)
        return s * (1-s)
    else:
        # A = np.where(Z > 0, 1. / (1. + np.exp(-Z)), np.exp(Z) / (np.exp(Z) + np.exp(0)))
        A = expit(Z)
        cache = Z
        return A, cache

def tanh(Z, derivative=False):
    """
    Implemented tanh and derivative of tanh for a single activation computation

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- Output of the linear layer, of any shape.  Also called cache
    derivative -- bool to determine if we are computing derivative or not 

    Returns:
    if derivative == False:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    if derivative == True:
        dZ -- Gradient of the cost with respect to Z
    """

    if derivative:
        return (1. - np.square(np.tanh(Z)))
    else:
        A = np.tanh(Z)
        cache = Z
        return A, cache

def relu(Z, derivative=False):
    """
    Implement the RELU function or implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- Output of the linear layer, of any shape.  Also called cache
    derivative -- bool to determine if we are computing derivative or not 

    Returns:
    if derivative == False:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    if derivative == True:
        dZ -- Gradient of the cost with respect to Z
    """
    if derivative:
        # dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        # # When z <= 0, you should set dz to 0 as well. 
        # dZ[Z <= 0] = 0
        # return dZ
        deriv = Z
        deriv[deriv <= 0] = 0
        deriv[deriv > 0] = 1
        return deriv
    else:
        A = np.maximum(0,Z)
        cache = Z 
        return A, cache
