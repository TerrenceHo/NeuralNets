import numpy as np

"""
Defines activation functions for connected layers in a network
sigmoid, tanh, and relu include derivatives
Softmax does not include derivatives, because it should only be used for the
last layer since it is numerically unsable
"""

def softmax(self, w):
    """ TODO: compute derivative
    Computes vector wise softmax NOTE: has no derivative
    """
    # logC = -np.max(w)
    w -= np.max(w)
    return np.exp(w)/np.sum(np.exp(w), axis=0)

def sigmoid(self, w, derivative=False):
    """
    Computes vector wise sigmoid or its derivative
    """
    d = 1.0/(1.0 + np.exp(-w))
    if derivative:
        return d * (1 - d)
    return d

def tanh(self, w, derivative=False):
    """
    Computes vector wise tanh or its derivative
    """
    if derivative:
        return 1 - np.square(np.tanh(w))
    return np.tanh(w)

def relu(self, w, derivative=False):
    """
    Computes vector wise relu func or its derivative
    """
    if derivative:
        deriv = w
        deriv[deriv <= 0] = 0
        deriv[deriv > 0] = 1
        return deriv
    return np.maximum(w, 0)

