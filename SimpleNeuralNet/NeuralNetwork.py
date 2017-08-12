# Std Libs
import numpy as np
import matplotlib.pyplot as plt

# Custom Libs
from activations import *
from cost_functions import *
from layers import *


class NeuralNetwork(object):
    """
    Class that holds Neural Network parameters
    """

    def __init__(self):
        """ 
        Initializes Neural Network Object

        Fields:
        parameters -- dictionary of weights and biases
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        costs -- list of all computed costs
        """

        self.parameters = {}
        self.costs = []

    def initialize_parameters(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3) # keep same parameters
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def Fit(self, X, Y, layer_dims, learning_rate = 0.0075,
            num_iterations=3000, print_cost = False):
        """
        Function that fits weights to the dataset X and labels Y given.

        Arguments:
        X -- dataset to train with and label
        Y -- labels for each example in dataset
        layer_dims -- list of layer sizes for each hidden layer. Size of list
            determines number of layers for neural network
        learning_rate -- step rate at which network updates its parameters.
            Default: 0.0075
        num_iterations -- How many epochs or iterations to train over dataset.
            Default: 3000
        print_cost -- Bool to decide whether or not to print cost. Default: False
        """

        self.parameters = self.initialize_parameters(layer_dims)
        self.GradientDescent(X, Y, learning_rate, num_iterations, print_cost)


    def GradientDescent(self, X, Y, learning_rate = 0.0075, num_iterations =
            3000, print_cost = False):
        """
        Applies Gradient Descent to update parameters and lower cost function

        Arguments:
        X -- dataset to train with and label
        Y -- labels for each example in dataset
        learning_rate -- step rate at which network updates its parameters.
            Default: 0.0075
        num_iterations -- How many epochs or iterations to train over dataset.
            Default: 3000
        print_cost -- Bool to decide whether or not to print cost. Default: False
        """

        for i in range(num_iterations):
            AL, caches = L_model_forward(X, self.parameters)
            cost = Cross_Entropy_Loss(AL, Y)
            grads = L_model_backward(AL, Y, caches)
            self.parameter = self.Update_Parameters(self.parameters, grads, learning_rate)

            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.costs.append(cost)

    def Update_Parameters(self, parameters, grads, learning_rate):
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

    def Predict(self, X):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))

        # Forward propagation
        probas, caches = L_model_forward(X, self.parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p

    def Score(self, X, Y, print_accuracy=True):
        """
        This function is used to computer accuracy of the neural network

        Arguments:
        X -- dataset of test data you would like to predict and label
        Y -- dataset of the test labels you would like to predict on
        print_accuracy -- bool to print out computed accuracy. Default: True

        Returns:
        Y_Pred -- Predictions for given dataset X
        Score -- Score for given dataset X and labels Y
        """
        m = X.shape[1]

        # Get Predictions
        Y_pred = self.Predict(X)
        Score = np.sum((Y_pred == Y)/m)

        if print_accuracy:
            print("Accuracy: "  + str(Score))
        return Y_pred, Score

    def Graph_Costs_Over_Time(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
