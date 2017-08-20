# Std Libs
import numpy as np
import matplotlib.pyplot as plt

# Custom Libs
from optimization_functions import *


class NeuralNetwork(object):
    """
    Class that holds Neural Network parameters
    """

    def __init__(self, init_method, layers_dims, keep_probs,
            optimization_function, activation_funcs,
            cost_func, reg_type, reg_lambd=0.1, learning_rate = 0.0075):
        """ 
        Initializes Neural Network Object

        Arguments:
        init_method -- method used to initialize_parameters.  Can be string or
            int.  Default = 0.01
        layers_dims -- list containing dimensions of each layer
        keep_probs -- list of floats to determine which neurons to shut down
        optimization_function -- function to optimize neural network
        activation_funcs -- non-linear functions to activate weights
        cost_func -- cost function to evaluate performance
        reg_type -- function that returns a regularization function
        reg_lambd -- lambd value that determines rate of regularization
        learning_rate -- rate at which optimization function will change
            parameters. Default = 0.0075

        Additional Fields:
        parameters -- dictionary of weights and biases
                        Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                        bl -- bias vector of shape (layers_dims[l], 1)
        costs -- list of all computed costs
        """

        self.parameters = self.Initialize_Parameters(layers_dims, init_method)
        self.costs = []
        self.layers_dims = layers_dims
        self.optimization_function = optimization_function
        self.activation_funcs = activation_funcs
        self.cost_func = cost_func
        self.reg_type = reg_type
        self.reg_lambd = reg_lambd
        self.learning_rate = learning_rate
        self.keep_probs = keep_probs

    def Initialize_Parameters(self, layers_dims, method=0.01):
        """
        Initializes parameters randomly and scales it by var method.  If method
        == 'He', scale = 2.0/np.sqrt(layers_dims[l-1]).  If method == 'Xavier',
        scale = 1.0/np.sqrt(layers_dims[l-1].  Else the default is 0.01

        Arguments:
        layers_dims -- python array (list) containing the dimensions of each layer in our network
        method -- string or int dictating how the parameters are initialzed.
        Default = 0.01

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layers_dims[l], layers_dims[l-1])
                        bl -- bias vector of shape (layers_dims[l], 1)
        """

        np.random.seed(1) # keep same parameters
        parameters = {}
        L = len(layers_dims)            # number of layers in the network

        for l in range(1, L):
            if isinstance(method, str):
                if method == 'He':
                    scale = 2.0/np.sqrt(layers_dims[l-1])
                elif method == 'Xavier':
                    scale = 1.0/np.sqrt(layers_dims[l-1])
                else:
                    raise AttributeError("Initialization method not found")
                    return
            else:
                scale = method
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * scale
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters


    def Fit(self, X, Y, num_iterations, print_cost=False):
        """
        Function that fits weights to the dataset X and labels Y given.

        Arguments:
        X -- dataset to train with and label
        Y -- labels for each example in dataset
        num_iterations -- number of iterations to train for
        print_cost -- Bool to decide whether or not to print cost. Default: False
        """

        # Check dimensions before fitting to optimization function
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Number of examples in dataset and labels do not match")
            return
        elif X.shape[0] != self.layers_dims[0]:
            raise ValueError("Input features in dataset do not match previously given input layer dimensions")
            return
        elif Y.shape[0] != self.layers_dims[-1]:
            raise ValueError("Output layer dimensions do not match previously given output layer dimensions")

        self.parameters, self.costs = self.optimization_function(X, Y, self.parameters,
                self.costs, self.activation_funcs, self.keep_probs, self.cost_func,
                self.reg_type, self.reg_lambd, self.learning_rate,
                num_iterations, print_cost)

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
        keep_probs = [1.0 for keep_prob in self.keep_probs]

        # Forward propagation
        probas, caches = L_model_forward(X, self.parameters,
                self.activation_funcs, keep_probs)

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
        """
        This function takes the list of costs saved in the Neural Network object
        and plots them onto a graph
        """
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
