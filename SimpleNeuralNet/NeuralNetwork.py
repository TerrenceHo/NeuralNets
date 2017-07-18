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
                 minibatch=1, epochs=250, dropout=False, dropout_rate=0.0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.l2 = l2
        self.momentum_const = momentum_const
        self.minibatch = minibatch
        self.epochs = 250
        self.dropout = dropout
        self.dropout_rate = dropout_rate

    def init_weights(self):
        """
        initializes thetas with random but uniform weights between -1 and 1
        """
        w1 = np.random.uniform(-1.0, 1.0, size=(
            self.n_hidden * (self.input_size + 1))).reshape(
                self.n_hidden, self.input_size + 1)

        w2 = np.random.uniform(-1.0, 1.0, size=(
            self.output_size * (self.hidden_size + 1))).reshape(
                self.output_size, self.hidden_size + 1)

        return w1, w2

    def one_hot(self, y, size):
        onehot = np.zeros((size, y.shape[0]))
        for i in range(y.shape[0]):
            onehot[y[i], i] = 1.0
        return y

    # def forwardProp():

    # def backProp():

    # def getCost():

    # def fit():
