from processing.vector_transform import *
from cost_functions import *
from layers import *

import numpy as np


def gradient_check(parameters, gradients, X, Y, activation_funcs, cost_func,
        reg_func, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, keys = dictionary_to_vector(parameters)
    grad, _ = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    keep_probs = [1.0 for i in activation_funcs]

    # Compute gradapprox
    for i in range(num_parameters):

        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] += epsilon
        thetaplus_vec = vector_to_dictionary(thetaplus, keys)
        AL, caches = L_model_forward(X, thetaplus_vec, activation_funcs, keep_probs)
        J_plus[i] = cost_func(AL, Y, reg_func, thetaplus_vec)
        ### END CODE HERE ###

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] -= epsilon 
        thetaminus_vec = vector_to_dictionary(thetaminus, keys)
        AL, caches = L_model_forward(X, thetaminus_vec, activation_funcs, keep_probs)
        J_minus[i] = cost_func(AL, Y, reg_func, thetaminus_vec)
        ### END CODE HERE ###

        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
        ### END CODE HERE ###

    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox) 
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator/denominator 
    ### END CODE HERE ###

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

def Check(X, Y, parameters, activation_funcs, cost_func):
    reg_func = L2_Regularization(0.0)
    keep_probs = [1.0 for i in activation_funcs]

    AL, caches = L_model_forward(X, parameters, activation_funcs, keep_probs)
    cost = cost_func(AL, Y, reg_func, parameters)
    grads = L_model_backward(AL, Y, caches, cost_func, reg_func)

    difference = gradient_check(parameters, grads, X, Y, activation_funcs,
            cost_func, reg_func)
