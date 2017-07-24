import numpy as np
import sys
from random import shuffle
from activations import *

class NeuralNetwork(object):
    """
    layer_sizes: list of number of nodes in each layer.  Ex: [50, 15, 5]  The first
        layer is assumed to be the input layer, and the last is the hidden layer.
        Everything else represents the size of the hidden_layer
    learning_rate: rate at which neural network adjusts its gradients (default: 0.01)
    decay_rate: rate at which learning rate decreases (default: 0.0)
    l2: l2 regularization factor (default: 0.0)
    momentum_const: multiply with the gradient of the previous pass through (default: 0.0)
    minibatch: size of smaller subsets of data, for efficiency (default: 1)
    epochs: number of passes through dataset (default: 250)
    dropout: If True, then neuralnet drops dropout_rate % of weights (default: False)
    dropout_rate: % of which weights to dropout (default: 0.0)
    
    weights: is a list that holds each weight

    TODO
    nesterov
    different forms of optimizers
    """

    def __init__(self, layer_sizes,
                 learning_rate=0.01, decay_rate=0.0, l2=0.0, momentum_const=0.0,
                 minibatch=1, epochs=250, dropout=False, dropout_rate=0.0,
                 check_gradients=False):
        # Hyper parameters
        self.layer_sizes = layer_sizes
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
        self.weights = self.init_weights()

    def init_weights(self):
        """
        initializes thetas with random but uniform weights between -1 and 1
        """
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.uniform(-1.0, 1.0, size=(
                (self.layer_sizes[i] + 1) *self.layer_sizes[i+1])).reshape(
                        self.layer_sizes[i] + 1, self.layer_sizes[i + 1])
            weights.append(w)

        return weights

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


    def get_cost(self, y_enc, output):
        """ Compute the cost function.
            Params:
            y_enc: array of num_labels x num_samples. class labels one-hot encoded
            output: matrix of output_units x samples - activation of output layer from feedforward
            w1: weight matrix of input to hidden layer
            w2: weight matrix of hidden to output layer
            """
        cost = -np.sum(y_enc*np.nan_to_num(np.log(output)))
        # if np.isnan(cost) == True:
        #     sys.exit()

        # add the L2 regularization by taking the L2-norm of the weights and multiplying it with our constant.
        l2_term = 0
        # (self.l2/2.0) * (np.sum(np.square(w1[:, 1:])) + np.sum(np.square(w2[:, 1:])))
        for i in range(len(self.weights)):
            l2_term += (self.l2/2.0) * (np.sum(np.square(self.weights[i][1:,:])))

        cost += l2_term
        return cost/y_enc.shape[0]

    def forwardProp(self, X, dropout = True):
        """
        X: input np array
        w1: matrix of weights from input layer to hidden layer
        w2: matrix of weights from hidden layer to output layer
        dropout: If true, will apply dropout_rate to each layer
        Feeds forwards the inputs to output layer and then returns both weights
        all three layers for backPropagation later
        """
        # lists to hold out activated layers and z vectors
        activations, zs = [], []
        a = X
        for i in range(len(self.weights)):
            # Insert a bias, which is the activation for the input layer
            a = self.add_bias_unit(a)
            activations.append(a)
            # compute dropout
            if self.dropout and dropout: a = self.compute_dropout(a)
            # apply weights to inputs for a linear transformation
            z = a.dot(self.weights[i])
            zs.append(z)
            # Activation function which maps values between 0 and 1
            a = sigmoid(z)
        activations.append(a)

        return activations, zs

    def backProp(self, activations, zs, y_enc):
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
        grads = []
        sigma = activations[-1] - y_enc
        grad = activations[-2].T.dot(sigma)
        # grad = sigma.dot(activations[-2].T)
        grads.insert(0,grad)

        for i in range(2, len(self.layer_sizes)):
            sigma = sigma.dot(self.weights[-i + 1][1:,:].T) * sigmoid(zs[-i], derivative=True)
            grad = activations[-i -1].T.dot(sigma)
            grads.insert(0, grad)

        for i in range(len(grads)):
            grads[i][1:,:] += (self.weights[i][1:,:]*self.l2)

        return grads
        # #backPropagate our error
        # # tells us the direction to move in
        # sigma3 = a3 - y_enc
        # sigma2 = sigma3.dot(w2[:,1:]) * sigmoid(z2, derivative=True)

        # # calculate the amounts needed to move
        # grad2 = sigma3.T.dot(a2)
        # grad1 = sigma2.T.dot(a1)
        # # add the regularization term
        # grad1[:, 1:]+= (w1[:, 1:]*self.l2) # derivative of .5*l2*w1^2
        # grad2[:, 1:]+= (w2[:, 1:]*self.l2) # derivative of .5*l2*w2^2
        # return grad1, grad2

    def fit(self, X, y, print_progress=True, check_gradients=True):
        """ Learn weights from training data
            Params:
            X: matrix of samples x features. Input layer
            y: target class labels of the training instances (ex: y = [1, 3, 4, 4, 3])
            print_progress: True if you want to see the loss and training accuracy, but it is expensive.
        """
        X_data, y_data = X.copy(), y.copy()

        y_enc = self.one_hot(y_data, self.layer_sizes[-1])
        X_split = np.array_split(X_data, self.minibatch)
        y_split = np.array_split(y_enc, self.minibatch)

        # PREVIOUS GRADIENTS
        prev_grads = []
        for i in range(len(self.weights)):
            prev_grads.append(np.zeros(self.weights[i].shape))

        #pass through the dataset
        for epoch in range(self.epochs):
            self.learning_rate /= (1 + self.decay_rate*epoch)
            for i in range(len(X_split)): # Feed each minibatch
                #feed feedforward
                activations, zs = self.forwardProp(X_split[i], dropout=True)
                cost = self.get_cost(y_split[i], output=activations[-1])

                #compute gradient via backpropagation
                grads = self.backProp(activations, zs, y_enc=y_split[i])

                # Check Gradients
                if check_gradients:
                    self.check_num_gradients(X_split[i], grads[0])

                # update parameters, multiplying by learning rate + momentum constants
                weight_updates = []
                for i in range(len(grads)):
                    weight_updates.append(self.learning_rate * grads[i])

                # w1_update = self.learning_rate*grad1
                # w2_update = self.learning_rate*grad2

                # gradient update: w += -alpha * gradient.
                # use momentum - add in previous gradient mutliplied by a momentum hyperparameter.
                for i in range(len(self.weights)):
                    self.weights[i] += -(weight_updates[i] + (self.momentum_const * prev_grads[i]))
                prev_grads = weight_updates
                # self.w1 += -(w1_update + (self.momentum_const*prev_grad_w1))
                # self.w2 += -(w2_update + (self.momentum_const*prev_grad_w2))
                # prev_grad_w1, prev_grad_w2 = w1_update, w2_update

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

    def check_num_gradients(self, X_data, grad_w1):
        h = 1e-5
        _, _, _, _, out1 = self.forwardProp(X_data, self.w1 + h, self.w2, dropout=False)
        _, _, _, _, out2 = self.forwardProp(X_data, self.w1 - h, self.w2, dropout=False)
        num_grad = (out1 - out2)/float(2 * h)
        num_grad = np.sum(num_grad)
        analytical_grad = np.sum(grad_w1)
        error = np.abs(analytical_grad -
                num_grad)/max(np.abs(analytical_grad), np.abs(num_grad))
        print("Gradient Error: {}".format(error))


    def predict(self, X, dropout = False):
        """
        X: matrix of training data, with dimensions of samples X input_size

        After training a model with fit, predict with those same weights it
        just learned with
        Returns matrix of predictions of highest probablility
        """
        activations, _ = self.forwardProp(X, dropout = False)
        pred = np.argmax(activations[-1], axis = 1)
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


