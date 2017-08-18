import numpy as np
from activations import *

# ===== Forward Layers ===== 
def dropout_forward(A, keep_prob):
    """
    Implements dropout for one activated layer

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    keep_prob -- float from 0.0 to 1.0 that determines how much of a layer to keep

    Returns:
    A -- activations after applying dropout_layer
    dropout_cache -- list containing dropout_layer, keep_prob
        dropout_layer -- vector of A.shape, which determines which activations
            from A to drop
        keep_prob -- probability of activations to keep
    """

    dropout_layer = np.random.rand(A.shape[0], A.shape[1])
    dropout_layer = dropout_layer < keep_prob
    A *= dropout_layer
    A /= keep_prob

    dropout_cache = (dropout_layer, keep_prob)
    return A, dropout_cache

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation, keep_prob):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a function
    keep_prob -- probability of the layer to be kept during dropout

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python list containing "linear_cache", "activation_cache",
        activation function, and D(dropout layer);
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation(None, Z)

    # Dropout
    A, dropout_cache = dropout_forward(A, keep_prob)

    cache = (linear_cache, activation_cache, activation, dropout_cache)

    return A, cache

def L_model_forward(X, parameters, activation_funcs, keep_probs):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- weights to pass forward
    activation_funcs -- list of activation functions
    keep_probs -- list of probabilities of to keep during dropout
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)],
                parameters['b' + str(l)], activation_funcs[l-1], keep_probs[l-1])
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],
            parameters['b' + str(L)], activation_funcs[L-1], 1.0)
    caches.append(cache)
    
    return AL, caches


# ===== Backwards Layers =====

def dropout_backwards(dA, dropout_cache):
    """
    Applies dropout layer to the same activated to shut down the same neurons as
    during forward propagation

    Arguments:
    dA -- derivative of the activated layer
    dropout_cache -- list containing dropout_layer, keep_prob
        dropout_layer -- vector of A.shape, which determines which activations
            from A to drop
        keep_prob -- probability of activations to keep

    Returns:
    dA -- derivative layer with dropout applied onto it
    """

    dropout_layer, keep_prob = dropout_cache
    dA *= dropout_layer
    dA /= keep_prob
    return dA

def linear_backward(dZ, cache, reg_function):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    reg = reg_function(W, derivative=True)/m

    dW = 1./m * np.dot(dZ,A_prev.T) + reg
    db = 1./m * np.sum(dZ)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, reg_function):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache, activation, dropout_cache = cache
    dA = dropout_backwards(dA, dropout_cache)
    dZ = activation(dA, activation_cache, derivative=True)

    dA_prev, dW, db = linear_backward(dZ, linear_cache, reg_function)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, cost_function, reg_function):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    cost_function -- function to evaluate cost
    reg_function -- function to apply regularization to gradients
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation for cross_entropy_loss
    dAL = cost_function(AL, Y, None, None, derivative=True)
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = \
    linear_activation_backward(dAL, current_cache, reg_function)
    
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" +
            str(l + 2)], current_cache, reg_function)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

