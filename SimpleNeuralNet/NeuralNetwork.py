import numpy as np
from activations import *

class NeuralNetwork:
    """
    Class that holds neural network parameters
    """

    def __init__(self, n_x, n_h, n_y, learning_rate):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.parameters = self.__init_random_weights(n_x, n_h, n_y)

    def __init_random_weights(n_x, n_h, n_y):
        w1 = np.random.rand(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        w2 = np.random.rand(n_y, n_h) * 0.01
        b1 = np.zeros((n_y, 1))
        parameters =  {
                "W1":w1,
                "B1":b1,
                "W2":w2,
                "B2":b2
            }
        return parameters

    def GradientDescent(self, X, Y, learning_rate, epochs):
        for i in range(epochs):
            activations, cache = self.ForwardProp(X)
            cost = self.CrossEntropyLoss()
            grads = self.BackPropagation(activations, cache)

    def ForwardPropagation(self, X):
        return 0

    def BackPropagation(self, activations, cache):
        return 0

    def CrossEntropyLoss(self, Y_pred, Y):
        return 0

    def Predict(self, X):
        activations, cache = self.ForwardProp(X)
        A2 = activations[-1]
        A2[A2 >= 0.5] = 1
        A2[A2 < 0.5] = 0
        return A2

    def Score(self, X, Y):
        Y_pred = self.Predict(X)
        diffs = Y_pred - Y
        count = 0.0
        for i in range(y.shape[1]):
            if diffs[i] != 0:
                count += 1
        return 100 - count*100/y.shape[1]
