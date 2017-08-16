import numpy as np

from layers import *
from cost_functions import *

def GradientDescent(X, Y, parameters, costs, activation_funcs, cost_func,
        reg_type, reg_lambd=0.1, learning_rate = 0.0075, num_iterations =
        3000, print_cost = False):
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

    # Returns regularization function that will calculate regularization
    reg_func = reg_type(reg_lambd)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters, activation_funcs)
        cost = Cross_Entropy_Loss(AL, Y, reg_func, parameters)
        grads = L_model_backward(AL, Y, caches, activation_funcs, Cross_Entropy_Loss, reg_func)
        parameters = Update_Parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    return parameters, costs

def Update_Parameters(parameters, grads, learning_rate):
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

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters
