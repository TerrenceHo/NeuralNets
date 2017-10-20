import numpy as np
import math

from .layers import *
from .cost_functions import *

# dictionary to list default configurations for all updating algorithms
default_configs = {
        "seed":0,
        "reg_lambd": 0.0,
        "learning_rate": 0.0075,
        "num_iterations": 3000,
        "mini_batch_size":64,
        "print_cost":False
    }

def ClassicMomentum(X, Y, parameters, costs, activation_funcs, keep_probs,
        cost_func, reg_type, v, beta, **kwargs):

    # Pass in default configs
    configs = dict(default_configs)
    configs.update(kwargs)

    # define velocity parameters for each weight
    v = {}
    L = len(parameters) // 2
    for l in range(1, L):
        v["dW" + str(l)] = np.zeros(parameters["W" + l].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + l].shape)

    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters 
                        parameters["W" + str(l)] = ... 
                        parameters["b" + str(l)] = ...
        """
        for l in range(1, L):
            v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW"
                    + str(l)]
            v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db"
                    + str(l)]
            # update parameters

            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]
        return parameters

    parameters, costs = BaseUpdate(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func,
        reg_type, update_parameters, **configs)
    return parameters, costs


def RMSProp(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func,
        reg_type, beta1, s, t, beta2, **kwargs):

    # Pass in default configs
    configs = dict(default_configs)
    configs.update(kwargs)

    L = len(parameters) // 2
    s = {}
    
    for l in range(L):
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

    s_corrected = {}

    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters 
                        parameters["W" + str(l)] = ... 
                        parameters["b" + str(l)] = ...
        """

        for l in range(L):
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)] * grads["dW" + str(l+1)]
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)] * grads["db" + str(l+1)]

            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1 - beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1 - beta2**t)

            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
        return parameters

def AdamOptimizer(X, Y, parameters, costs, activation_funcs, keep_probs,
        cost_func, reg_type, beta1, beta2, epsilon, **kwargs):

    # Pass in default configs
    configs = dict(default_configs)
    configs.update(kwargs)

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

    v_corrected = {}
    s_corrected = {}

    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters 
                        parameters["W" + str(l)] = ... 
                        parameters["b" + str(l)] = ...
        """

        for l in range(L):
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta1**t)

            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)] * grads["dW" + str(l+1)]
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)] * grads["db" + str(l+1)]

            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1 - beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1 - beta2**t)

            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon))
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon))
        return parameters

    parameters, costs = BaseUpdate(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func,
        reg_type, update_parameters, **configs)
    return parameters, costs


def GradientDescent(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func,
        reg_type, **kwargs):
    """
    Applies Gradient Descent to update parameters and lower cost function

    Arguments:
    X -- dataset to train with and label
    Y -- labels for each example in dataset
    parameters -- dictionary of weights and biases
                    Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                    bl -- bias vector of shape (layers_dims[l], 1)
    costs -- list of all computed costs
    learning_rate -- step rate at which network updates its parameters.
        Default: 0.0075
    num_iterations -- How many epochs or iterations to train over dataset.
        Default: 3000
    print_cost -- Bool to decide whether or not to print cost. Default: False
    lambd -- regularization parameter, if zero there is not regularization.
        Default: 0.1
    method -- which regularization method to use. Default: l2

    Returns:
    parameters -- dictionary of weights and biases
                    Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                    bl -- bias vector of shape (layers_dims[l], 1)
    costs -- list of all computed costs
    """

    # Pass in default configs
    configs = dict(default_configs)
    configs.update(kwargs)

    # Define Update Parameters to pass into BaseUpdate function
    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters 
                        parameters["W" + str(l)] = ... 
                        parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2 # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters

    parameters, costs = BaseUpdate(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func,
        reg_type, update_parameters, **configs)
    return parameters, costs

def BaseUpdate(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func,
        reg_type, update_parameters, seed, reg_lambd, learning_rate, num_iterations,
        mini_batch_size, print_cost):
    """
    Base function for updating parameters

    Arguments:
    X -- dataset to train with and label
    Y -- labels for each example in dataset
    parameters -- dictionary of weights and biases
                    Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                    bl -- bias vector of shape (layers_dims[l], 1)
    costs -- list of all computed costs
    learning_rate -- step rate at which network updates its parameters.
        Default: 0.0075
    num_iterations -- How many epochs or iterations to train over dataset.
        Default: 3000
    print_cost -- Bool to decide whether or not to print cost. Default: False
    lambd -- regularization parameter, if zero there is not regularization.
        Default: 0.1
    method -- which regularization method to use. Default: l2

    Returns:
    parameters -- dictionary of weights and biases
                    Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                    bl -- bias vector of shape (layers_dims[l], 1)
    costs -- list of all computed costs
    """

    # Returns regularization function that will calculate regularization
    reg_func = reg_type(reg_lambd)

    for i in range(num_iterations):
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        seed += 1 # update differently each time
        for mini_batch in mini_batches:
            (mini_X, mini_Y) = mini_batch
            AL, caches = L_model_forward(mini_X, parameters, activation_funcs, keep_probs)
            cost = cost_func(AL, mini_Y) 
            cost += reg_func(parameters)/mini_Y.shape[1]

            grads = L_model_backward(AL, mini_Y, caches, cost_func)
            grads_reg = reg_func(parameters, derivative=True)
            grads = {k : grads.get(k,0) + grads_reg.get(k, 0) for k in
                    set(grads.keys()) | set(grads_reg.keys())}

            parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    return parameters, costs

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k+1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

