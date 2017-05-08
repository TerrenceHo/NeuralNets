from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from decimal import Decimal
import os


def main():
    # Variables
    input_size = 400 # size of image
    hidden_size = 25 # size of hidden layer
    num_labels = 10 # size of output later
    learning_rate = 1 # how much gradient decreases by
    iterations = 1000 # Put limit before stopping 

    #Load Data/Separate into pieces
    os.system("clear")
    print("Loading Data...")
    datafile = 'data.mat'
    mat = loadmat(datafile) # loads data
    X, y = mat['X'], mat['y'] # Get X, y values
    # y[y==10] = 0 # transform 10 to zero, easier prediction !!! doesn't work
    m = X[0].shape
    encoder = OneHotEncoder(sparse = False) # get matrix of y for comparison
    y_onehot = encoder.fit_transform(y)


    print("Load Initial Weights")
    input("Press Enter to continue...")

    # get neural net weights 
    Theta1 = randInitializeWeights(input_size, hidden_size)
    Theta2 = randInitializeWeights(hidden_size, num_labels)
    params = np.concatenate((np.ravel(Theta1), np.ravel(Theta2)))
    # params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels
    #                            * (hidden_size + 1)) - 0.5) * 0.25

    # Call minimize function to train neural network
    print("Start Neural Net Training")
    input("Press Enter to continue...")
    myArgs = (input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
    fmin = minimize(fun=nnCostFunction, x0=params, args= myArgs,
        method='TNC', jac=True, options={'maxiter': iterations})
    print(fmin)

    # get back the weights calculated from neural net 
    Theta1 = np.reshape(fmin.x[:hidden_size * (input_size + 1)],
                                  (hidden_size, (input_size + 1)))
    Theta2 = np.reshape(fmin.x[hidden_size * (input_size + 1):],
                                  (num_labels, (hidden_size + 1)))

    # Forward prop thorugh weights to compute training accuracy
    print("Predict Accuracy")
    input("Press Enter to continue...")
    a1, z2, a2, z3, h = forwardProp(X, Theta1, Theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('Accuracy = {0}%'.format(accuracy * 100))

    # Saving model to be used later
    savemat('model.mat', {'Theta1':Theta1, 'Theta2':Theta2})


def nnCostFunction(params, input_size, hidden_size, num_labels, X, y,
                   learning_rate):
    #reshape parameters that were flattened
    Theta1 = np.reshape(params[:hidden_size * (input_size + 1)],
                                  (hidden_size, (input_size + 1)))
    Theta2 = np.reshape(params[hidden_size * (input_size + 1):],
                                  (num_labels, (hidden_size + 1)))
    m = X.shape[0] # size of X 

    # Feed Forward Network
    a1, z2, a2, z3, h = forwardProp(X, Theta1, Theta2)

    # regularize terms
    Theta1Reg = np.sum(np.sum(np.square(Theta1[:,1:])) )
    Theta2Reg = np.sum(np.sum(np.square(Theta2[:,1:])) )
    # Theta1Reg = np.sum(np.square(Theta1[:,1:]))
    # Theta2Reg = np.sum(np.square(Theta2[:,1:]))

    r = (learning_rate/(2 * m)) * (Theta1Reg + Theta2Reg) # reg term for cost

    J = (1/m) * np.sum(np.sum((-y) * np.log(h) - (1-y) * np.log(1-h))) + r #cost

    print("Cost: %f" % J)

    # Starting Backpropagation
    # calculate deltas
    d3 = h - y
    d2 = sigmoidGradient(z2) * (d3.dot(Theta2[:,1:]))

    # calculate differences in weights
    Delta1 = d2.T.dot(a1)
    Delta2 = d3.T.dot(a2)

    # computing weights from backprob derivatives
    Theta1Grad = (1/m) * Delta1
    Theta2Grad = (1/m) * Delta2

    # regularizing the backprop
    Theta1[:,0] = 0
    Theta2[:,0] = 0

    # Getting new weights
    Theta1Grad = Theta1Grad + ((learning_rate/m) * Theta1)
    Theta2Grad = Theta2Grad + ((learning_rate/m) * Theta2)
    # return weights as one parameter
    grad = np.concatenate((np.ravel(Theta1Grad), np.ravel(Theta2Grad)))

    # Return both the weights and the cost
    return J, grad

def forwardProp(X, theta1, theta2):
    m = X.shape[0] # size of x
    a1 = np.insert(X, 0, values=np.ones(m), axis=1) #insert bias
    z2 = a1.dot(theta1.T) # Pass through first weights
    a2 = np.insert(expit(z2), 0, values=np.ones(m), axis=1) #insert bias
    z3 = a2.dot(theta2.T) # pass through hidden layer's weights
    h = expit(z3) # output
    return a1, z2, a2, z3, h

def sigmoidGradient(z): # function to calculate sigmoid for gradients
    d = expit(z)
    return d*(1-d)

def randInitializeWeights(L_in, L_out): # randomly distribute weights
    W = np.zeros((L_out, 1 + L_in)) # form weight vector
    epsilon_init = 0.12 # factor by which randomness is generated
    # generates weights
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init) - epsilon_init
    return W # return weights.


# The following functions have nothing to do with the neural net, but helped debug
# and see inside the neural net.

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10
    return W

def computeNumericalGradient(J, theta):
    numgrad = np.zeros( theta.shape )
    perturb = np.zeros( theta.shape )
    e = 1e-4

    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad

def checkGradients():
    lambda_reg = 0
    input_layer_size = 3
    hidden_layer_size = 5
    labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = np.array([[2], [3], [1], [2], [3]])

    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'),
                                Theta2.reshape(Theta2.size, order='F')))

    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, \
                   labels, X, y, lambda_reg)

    _, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    fmt = '{:<25}{}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

    print('The above two columns you get should be very similar.\n'
        '(Left Col.: Your Numerical Gradient, Right Col.: Analytical Gradient)')

    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))

    print('If your backpropagation implementation is correct, then \n' \
        'the relative difference will be small (less than 1e-9). \n' \
        '\nRelative Difference: {:.10E}'.format(diff))

if __name__ == '__main__':
    main() # run main to start neural net.











