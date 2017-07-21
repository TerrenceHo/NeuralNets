from activations import *
from random import shuffle
import numpy as np


class NeuralNetwork(object):
    """
    input_size: number of features in input dataset
    hidden_size: number of hidden layer units (default: 50)
    output_size: number of classes to predict on
    learning_rate: rate at which neural network adjusts its gradients (default: 0.01)
    decay_rate: rate at which learning rate decreases (default: 0.0)
    l2: l2 regularization factor (default: 0.0)
    momentum_const: multiply with the gradient of the previous pass through (default: 0.0)
    minibatch: size of smaller subsets of data, for efficiency (default: 1)
    epochs: number of passes through dataset (default: 250)
    dropout: If True, then neuralnet drops dropout_rate % of weights (default: False)
    dropout_rate: % of which weights to dropout (default: 0.0)

    TODO
    nesterov
    different forms of optimizers

    """
    def __init__(self, input_size, output_size, hidden_size=50,
                 learning_rate=0.01, decay_rate=0.0, l2=0.0, momentum_const=0.0,
                 minibatch=1, epochs=250, dropout=False, dropout_rate=0.0,
                 check_gradients=False):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.l2 = l2
        self.momentum_const = momentum_const
        self.minibatch = minibatch
        self.epochs = epochs
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.check_gradients = check_gradients

        self.w1 = np.zeros((2,3))
        self.w2 = np.zeros((2,3))

    def init_weights(self, L_in, L_out):
        """
        initializes thetas with random but uniform weights between -1 and 1
        """
        w = np.random.uniform(-1.0, 1.0, size=(
            (L_in + 1) * L_out)).reshape(L_in + 1, L_out)

        return w

    def one_hot(self, y, size):
        """
        y: np array of numbers
        size: number of classes inside y

        This method converts y into a onehot version of itself
        """
        onehot = np.zeros((size, y.shape[0]))
        for i in range(y.shape[0]):
            onehot[y[i], i] = 1.0
        return onehot.T


    def dropout_layer(self, layer):
        """
        layer:  np array which has already has had an activation function
            applied to it

        when applied, selects a percentage of layer to zero
        """
        drop_mask = np.random.binomial(1, self.dropout_rate,
                size=layer.shape).astype('uint8')
        layer *= drop_mask
        return layer

    def getCost(self, y, output, w1, w2):
        """
        y: one-hot encoded class labels for a section of data that was just
            passed through the forwardProp func
        outout: the probabilities calculated with  forwardProp
        w1: weights from input layer to hidden layer
        w2: weights from hidden layer to input layer

        Derives difference in output and y, and calculates a cost
        """

        m = y.shape[0]
        # L2 regularization
        w1Reg = np.sum(np.square(w1[:, 1:]))
        w2Reg = np.sum(np.square(w2[:, 1:]))
        r = 0.5 * self.l2 * (w1Reg + w2Reg)
        # Calculating loss and adding regularization (sigmoid cross entropy)
        J = -np.sum(y * np.log(output)) + r
        # print("cost ", J)
        return J/m

    def forwardProp(self, X, w1, w2, dropout=True):
        """
        X: input np array
        w1: matrix of weights from input layer to hidden layer
        w2: matrix of weights from hidden layer to output layer
        dropout: If true, will apply dropout_rate to each layer

        Feeds forwards the inputs to output layer and then returns both weights
        all three layers for backpropagation later
        """
        # Shape of array for bias
        m = X.shape[0]
        # insert a bias
        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        # dropout
        # if self.dropout and dropout:
        #     a1 = self.dropout_layer(a1)
        # multiply input + bias by weight1
        z2 = a1.dot(w1)
        # apply activation function
        a2 = tanh(z2)
        # add bias to the hidden layer
        a2 = np.insert(a2, 0, values=np.ones(m), axis=1)
        # dropout
        # if self.dropout and dropout:
        #     a2 = self.dropout_layer(a2)
        # multiply hidden layer + bias by weight 2
        z3 = a2.dot(w2)
        # apply activation function
        a3 = softmax(z3)
        return a1, z2, a2, z2, a3

    def backProp(self, a1, a2, a3, z2, y_onehot, w1, w2):
        """
        a1: array of num_samples * input_size + 1 (input activation)
        a2: array of num_samples * hidden_size + 1 (hidden activation)
        a3: array of num_samples * output_size + 1 (output activation)
        z2: input of hidden layer
        y_onehot: class labels encoded in onehot form
        w1: weight matrix of input to hidden layer
        w2: weight matrix of hidden to output layer

        Backpropagates the error back through the neural network
        """
        m = y_onehot.shape[0]

        # calculate difference between outputs and targets
        d3 = a3 - y_onehot
        # Calculate the difference in the hidden layer 
        d2 = tanh(z2, True) * d3.dot(w2.T[:, 1:])

        # calculate gradients
        grad1 = a1.T.dot(d2)
        grad2 = a2.T.dot(d3)

        # regularizing the gradient, but not the bias term
        grad1[1:, :] += (w1[1:, :] * self.l2)
        grad2[1:, :] += (w2[1:, :] * self.l2)

        # return gradients
        return grad1, grad2

    def fit(self, X, y, print_progress=False):
        """
        X: matrix of training data, with dimensions of samples X input_size
        y: array containing target data, [1,2,3,4]

        Trains a neural net with these inputs by learning weights
        """

        m = X.shape[0]
        y_onehot = self.one_hot(y, self.output_size)
        self.w1 = self.init_weights(self.input_size, self.hidden_size)
        self.w2 = self.init_weights(self.hidden_size, self.output_size)

        # split the data into mini-batches
        X_split = np.array_split(X, self.minibatch)
        y_split = np.array_split(y_onehot, self.minibatch)

        # Implementation of Gradient Descent
        for epoch in range(self.epochs):
            for i in range(len(X_split)):
                # feed forward and get back the activations
                a1, z2, a2, z3, a3 = self.forwardProp(X_split[i], self.w1, self.w2)
                # Get the cost
                cost = self.getCost(y_split[i], output=a3, w1=self.w1, w2=self.w2)
                # compute the gradients of the weights
                grad1, grad2 = self.backProp(a1, a2, a3, z2, y_split[i],
                                             self.w1, self.w2)
                if self.check_gradients:
                    # Check gradients
                    h = 1e-5
                    w1_h = self.w1 + h
                    _, _, _, _, out1 = self.forwardProp(X_split[i], w1_h, self.w2,
                            False)
                    w1_h = self.w1 - h
                    _, _, _, _, out2 = self.forwardProp(X_split[i], w1_h, self.w2,
                            False)
                    numerical_deriv_1 = (out1 - out2) /float(2 * h)
                    analytical = np.sum(grad1)
                    numerical = np.sum(numerical_deriv_1)
                    w1_grad_error = np.abs(analytical - numerical) / np.max(
                            np.abs(analytical), np.abs(numerical))
                    # print("Gradient Error: {}".format(w1_grad_error))


                # Remove the bias term of the weights
                self.w1[0, :] = 0
                self.w2[0, :] = 0
                # Update the weights  w += -alpha * gradient
                self.w1 += ((-self.learning_rate) * grad1)
                self.w2 += ((-self.learning_rate) * grad2)

            # Decay the learning Rate
            self.learning_rate = self.learning_rate / (1 + self.decay_rate * epoch)
            # Shuffle the mini-batches 
            combined = list(zip(X_split, y_split))
            shuffle(combined)
            X_split[:], y_split[:] = zip(*combined)

            if print_progress and (epoch + 1) % 25 == 0 or epoch < 25:
                accuracy = self.accuracy(X, y)
                print("=======================================================")
                print("Epoch: %d, Loss: %d, Accuracy: %d" %
                        (epoch + 1, cost, accuracy))

    def predict(self, X):
        """
        X: matrix of training data, with dimensions of samples X input_size

        After training a model with fit, predict with those same weights it
        just learned with
        Returns matrix of predictions of highest probablility
        """

        a1, z2, a2, z3, a3 = self.forwardProp(X, self.w1, self.w2, dropout=False)
        pred = np.argmax(a3, axis=1)
        return pred

    def accuracy(self, X, y):
        """
        Predicts the accuracy by passing X through neural net and computing a
        score

        X: input data
        y: target classes
        """
        y_pred = self.predict(X).reshape(-1, 1)
        diffs = y_pred - y
        count = 0
        for i in range(y_pred.shape[0]):
            if diffs[i] != 0:
                count += 1
        return 100 - ((count * 100)/y_pred.shape[0])
