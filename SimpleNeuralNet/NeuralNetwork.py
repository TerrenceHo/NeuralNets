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

        self.w1
        self.w2

    def init_weights(self, L_in, L_out):
        """
        initializes thetas with random but uniform weights between -1 and 1
        """
        w = np.random.uniform(-1.0, 1.0, size=(
            L_out * (L_in + 1))).reshape(L_out, L_in + 1)

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
        return y

    def sigmoid(self, w):
        return 1.0/(1 + np.exp(-w))

    def dropout_layer(self, layer):
        """
        layer:  np array which has already has had an activation function applied
            to it

        when applied, selects a percentage of layer to zero
        """
        drop_mask = np.random.binomial(1, self.dropout_rate, size=layer.shape)
        layer *= drop_mask
        return layer

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
        if self.dropout and dropout:
            a1 = self.dropout_layer(a1)
        # multiply input + bias by weight1
        z2 = a1.dot(w1.T)
        # apply activation function
        a2 = self.sigmoid(z2)
        # add bias to the hidden layer
        a2 = np.insert(a2, 0, values=np.ones(m), axis=1)
        # dropout
        if self.dropout and dropout:
            a2 = self.dropout_layer(a2)
        # multiply hidden layer + bias by weight 2
        z3 = a2.dot(w2.T)
        # apply activation function
        a3 = self.sigmoid(z3)
        return a1, z2, a2, z2, a3

    # def backProp():

    def getCost(self, y, output, w1, w2):
        """
        y: one-hot encoded class labels for a section of data that was just
            passed through the forwardProp func
        outout: the probabilities calculated with  forwardProp
        w1: weights from input layer to hidden layer
        w2: weights from hidden layer to input layer
        fdsalkjfsa

        Derives difference in output and y, and calculates a cost
        """

        m = y.shape[0]
        w1Reg = np.sum(np.square(w1[:, 1:]))
        w2Reg = np.sum(np.square(w2[:, 1:]))
        r = (self.learning_rate/(2*m)) * (w1Reg + w2Reg)
        J = (1/m) * np.sum( (-1 * y * np.log(output)) -
                     ((1-y) * np.log(1 - output) ) ) + r
        return J

    def fit(self, X, y):
        """
        X: matrix of training data, with dimensions of samples X input_size
        y: array containing target data, [1,2,3,4]

        Trains a neural net with these inputs by learning weights
        """

        X_data = X.copy()
        y_data = y.copy()
        y_onehot = self.one_hot(y_data, self.output_size)
        self.w1 = self.init_weights(self.input_size, self.hidden_size)
        self.w2 = self.init_weights(self.hidden_size, self.output_size)

        # split the data into mini-batches
        X_split = np.array_split(X_data, self.minibatch)
        y_split = np.array_split(y_onehot, self.minibatch)

        # Implementation of Gradient Descent
        for epoch in range(self.epochs):
            for i in range(len(X_split)):
                a1, z2, a2, z3, a3 = self.forward_prop(X_split[i], self.w1, self.w2)
                cost = self.get_cost(y_split[i], output=a3, w1=self.w1, w2=self.w2)
                grad1, grad2 = self.backProp()

    def predict(self, X):
        """
        X: matrix of training data, with dimensions of samples X input_size
        After training a model with fit, predict with those same weights it
        just learned with
        """
