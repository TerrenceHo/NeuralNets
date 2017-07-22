import numpy as np
from random import shuffle
from activations import *

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
        # Hyper parameters
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

        # initialize weights
        self.w1 = self.init_weights(self.input_size, self.hidden_size)
        self.w2 = self.init_weights(self.hidden_size, self.output_size)

    def init_weights(self, L_in, L_out):
        """
        initializes thetas with random but uniform weights between -1 and 1
        """
        w = np.random.uniform(-1.0, 1.0, size=(
            L_out * (L_in + 1) )).reshape(L_out, L_in+1)
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

    def add_bias_unit(self, X, column=True):
        """Adds a bias unit to our inputs"""
        if column:
            bias_added = np.ones((X.shape[0], X.shape[1] + 1))
            bias_added[:, 1:] = X
        else:
            bias_added = np.ones((X.shape[0] + 1, X.shape[1]))
            bias_added[1:, :] = X

        return bias_added

    def compute_dropout(self, activations):
        """
        Sets half of the activations to zero
        Params: activations - numpy array
        Return: activations, which half set to zero
        """
        mult = np.random.binomial(1, self.dropout_rate, size = activations.shape)
        activations*=mult
        return activations


    def get_cost(self, y_enc, output, w1, w2):
        """ Compute the cost function.
            Params:
            y_enc: array of num_labels x num_samples. class labels one-hot encoded
            output: matrix of output_units x samples - activation of output layer from feedforward
            w1: weight matrix of input to hidden layer
            w2: weight matrix of hidden to output layer
            """
        cost = - np.sum(y_enc*np.log(output))
        # add the L2 regularization by taking the L2-norm of the weights and multiplying it with our constant.
        l2_term = (self.l2/2.0) * (np.sum(np.square(w1[:, 1:])) + np.sum(np.square(w2[:, 1:])))
        cost = cost + l2_term
        return cost/y_enc.shape[1]

    def forwardProp(self, X, w1, w2, dropout = True):
        """
        X: input np array
        w1: matrix of weights from input layer to hidden layer
        w2: matrix of weights from hidden layer to output layer
        dropout: If true, will apply dropout_rate to each layer
        Feeds forwards the inputs to output layer and then returns both weights
        all three layers for backPropagation later
        """
        # Insert a bias, which is the activation for the input layer
        a1 = self.add_bias_unit(X)
        # compute dropout
        if self.dropout and dropout: a1 = self.compute_dropout(a1)
        # apply weights to inputs for a linear transformation
        z2 = a1.dot(w1.T)
        # Activation function which maps values between 0 and 1
        a2 = tanh(z2)
        #add a bias unit to activation of the hidden layer.
        a2 = self.add_bias_unit(a2)
        # dropout
        if self.dropout and dropout: a2 = self.compute_dropout(a2)
        # apply weights to the hidden layer for a linear transformation
        z3 = a2.dot(w2.T)
        # the activation of our output layer is just the softmax function.
        # activation function for the output layer
        a3 = softmax(z3)
        return a1, z2, a2, z3, a3

    def backProp(self, a1, a2, a3, z2, y_enc, w1, w2):
        """
        a1: array of num_samples * input_size + 1 (input activation)
        a2: array of num_samples * hidden_size + 1 (hidden activation)
        a3: array of num_samples * output_size + 1 (output activation)
        z2: input of hidden layer
        y_onehot: class labels encoded in onehot form
        w1: weight matrix of input to hidden layer
        w2: weight matrix of hidden to output layer
        backPropagates the error back through the neural network
        """
        #backPropagate our error
        sigma3 = a3 - y_enc
        sigma2 = sigma3.dot(w2[:,1:]) * tanh(z2, derivative=True)
        #get rid of the bias row
        grad1 = sigma2.T.dot(a1)
        grad2 = sigma3.T.dot(a2)
         # add the regularization term
        grad1[:, 1:]+= (w1[:, 1:]*self.l2) # derivative of .5*l2*w1^2
        grad2[:, 1:]+= (w2[:, 1:]*self.l2) # derivative of .5*l2*w2^2
        return grad1, grad2


    def fit(self, X, y, print_progress=True):
        """ Learn weights from training data
            Params:
            X: matrix of samples x features. Input layer
            y: target class labels of the training instances (ex: y = [1, 3, 4, 4, 3])
            print_progress: True if you want to see the loss and training accuracy, but it is expensive.
        """
        X_data, y_data = X.copy(), y.copy()

        y_enc = self.one_hot(y_data, self.output_size)
        X_split = np.array_split(X_data, self.minibatch)
        y_split = np.array_split(y_enc, self.minibatch)
        # PREVIOUS GRADIENTS
        prev_grad_w1 = np.zeros(self.w1.shape)
        prev_grad_w2 = np.zeros(self.w2.shape)

        #pass through the dataset
        for epoch in range(self.epochs):
            self.learning_rate /= (1 + self.decay_rate*epoch)
            for i in range(len(X_split)): # Feed each minibatch
                #feed feedforward
                a1, z2, a2, z3, a3 = self.forwardProp(X_split[i], self.w1, self.w2)
                cost = self.get_cost(y_split[i], output=a3, w1=self.w1, w2=self.w2)

                #compute gradient via backpropagation
                grad1, grad2 = self.backProp(a1=a1, a2=a2, a3=a3, z2=z2,
                        y_enc=y_split[i], w1=self.w1, w2=self.w2)
                # update parameters, multiplying by learning rate + momentum constants
                w1_update = self.learning_rate*grad1
                w2_update = self.learning_rate*grad2
                # gradient update: w += -alpha * gradient.
                # use momentum - add in previous gradient mutliplied by a momentum hyperparameter.
                self.w1 += -(w1_update + (self.momentum_const*prev_grad_w1))
                self.w2 += -(w2_update + (self.momentum_const*prev_grad_w2))
                prev_grad_w1, prev_grad_w2 = w1_update, w2_update

            # Shuffle list after each epoch, to keep it random
            combined = list(zip(X_split, y_split))
            shuffle(combined)
            X_split[:], y_split[:] = zip(*combined)
            #print progress
            if print_progress and (epoch+1) % 50==0:
                print("Epoch: " + str(epoch+1))
                print("Loss: " + str(cost))
                acc = self.accuracy(X_data, y_data)
                print("Training Accuracy: " + str(acc))

        return self

    def predict(self, X, dropout = False):
        """
        X: matrix of training data, with dimensions of samples X input_size

        After training a model with fit, predict with those same weights it
        just learned with
        Returns matrix of predictions of highest probablility
        """
        a1, z2, a2, z3, a3 = self.forwardProp(X, self.w1, self.w2, dropout = False)
        pred = np.argmax(a3, axis = 1)
        return pred

    def accuracy(self, X, y):
        """
        Predicts the accuracy by passing X through neural net and computing a
        score

        X: input data
        y: target classes
        """
        y_pred = self.predict(X)
        diffs = y_pred - y
        count = 0.0
        for i in range(y.shape[0]):
            if diffs[i] != 0:
                count+=1
        return 100 - count*100/y.shape[0]


