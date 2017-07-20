import numpy as np
import NeuralNetwork
from load import *


def main():
    X_train, y_train = load_mnist('./data')
    X_test, y_test = load_mnist('./data', 't10k')

    parameters = {
            'input_size': X_train.shape[1],
            'hidden_size': 50,
            'output_size': 10,
            'learning_rate': 0.003,
            'decay_rate': 0.00001,
            'l2': 0.1,
            'minibatch': 500,
            'epochs': 5000,
            'dropout': True,
            'dropout_rate': 0.5,
        }
    nn = NeuralNetwork.NeuralNetwork(**parameters)
    nn.fit(X_train, y_train)
    accuracy = nn.accuracy(X_test, y_test)
    print(accuracy)


if __name__ == '__main__':
    main()
