import numpy as np

"""
Defines activation functions for connected layers in a network
sigmoid, tanh, and relu include derivatives
Softmax does not include derivatives, because it should only be used for the
last layer since it is numerically unsable
"""
def softmax(w):
    """Calculates the softmax function that outputs a vector of values that sum to one.
        We take max(softmax(v)) to be the predicted label. The output of the softmax function
        is also used to calculate the cross-entropy loss
    """
    logC = -np.max(w)
    return np.exp(w + logC)/np.sum(np.exp(w + logC), axis = 0)

def tanh(w, derivative=False):
    """
    Computes vector wise tanh or its derivative
    """
    if derivative:
        return 1 - np.square(np.tanh(w))
    return np.tanh(w)

def sigmoid(w, derivative=False):
    """
    Computes vector wise sigmoid or its derivative
    """
    d = 1.0/(1.0 + np.exp(-w))
    if derivative:
        return d * (1 - d)
    return d

def relu(w, derivative=False):
    """
    Computes vector wise relu func or its derivative
    """
    if derivative:
        deriv = w
        deriv[deriv <= 0] = 0
        deriv[deriv > 0] = 1
        return deriv
    relud = w
    relud[relud < 0] = 0
    return relud

