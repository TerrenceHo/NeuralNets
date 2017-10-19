# SimpleNeuralNet

__NOTE__: _This library is still a WIP_

This is an Python implementation of a fully connected neural network in NumPy. By using the matrix approach to neural networks, this NumPy implementation is able to harvest the power of the BLAS library and efficiently perform the required calculations. The network can be trained by a wide range of learning algorithms.

This library is intended to be expanded for personal use.  One could define their own learning algorithms, activation functions, or cost functions, so long as they follow a defined schema, which are defined below.

### Requirements
[NumPy](http://www.numpy.org/)

### Installation
When library is finished, this will be pip installable.  Until then, you can try out this library by cloning this repository into your source directory and using it there.

```
git clone https://github.com/TerrenceHo/NeuralNets.git
```

You will have to navigate to the SimpleNeuralNets repository.

### Implemented Learning Algorithms
- Gradient Descent
- Classic Momentum
- RMSProp
- Adam(Which is Momentum + RMSProp)

### Features
- Dropout
- Builtin Cost, Activation,  and Optimization Functions
- Batch Normalization

### Examples
One can see the examples of how to run this library 
[here](https://github.com/TerrenceHo/NeuralNets/blob/master/SimpleNeuralNet/RunNeuralNetwork.ipynb)

### Defining Your Own Functions
#### Activation Functions

```
def function_name(Z, derivative=False):
    if derivative:
        s = derivative_of_activation(Z)
        return s
    else:
        A = your_activation_here(Z)
        cache = Z
        return A, cache
```

-Z is the input layer to be activated or to calculate the derivative of an activated layer.  
-derivative is a boolean value to determine if needed to calculate the derivative of an activation function.

Activation functions are duo-fold.  They calculate both the forward pass as well as the derivative of an activation for backpropagation.  For the forward pass the function returns both the activated layer as well as the previous input layer.  

The following is an example of an implemented function.
```
def relu(Z, derivative=False):    
    if derivative:
        deriv = Z
        deriv[deriv <= 0] = 0
        deriv[deriv > 0] = 1
        return deriv
    else:
        A = np.maximum(0,Z)
        cache = Z
        return A, cache
```

#### Optimization Functions
```
def function_name(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func, reg_type, **kwargs):

    # Pass in default configs
    configs = dict(default_configs)
    configs.update(kwargs)
    
    # Optionally define other variables here to be operated on 
    # when update_parameters is running
    def update_parameters(parameters, gradients, learning_rate):
        L = len(parameters) // 2
        for l in range(L):
            # Operate on those previously defined optional variables.
            
            # Below is the basic code to update neural network weights and parameters.  
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
        return parameters
       
    parameters, costs = BaseUpdate(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func, reg_type, update_parameters, **configs)
    return probs, costs

```

- X is your training set of data
- Y is the set of answers corresponding to your answers
- parameters are the neural network weights.
- costs is a list of previous costs calculated
- activation_funcs is a list of activations functions to use
- keep_probs is a list of probabilities of nodes to keep every layer
- cost_func is the cost function to evaluate your model
- reg_type is a function that defines the regularization for your code

_Note that in your defined optimization function, you do not have to actually care about any of these variables above.  It is not necessary to worry about the arguments that are passed into kwargs, these are base arguments that are when the neural net is instantiated.  As long as the configs dictionary is passed in the kwargs the following process should work._

That looks like a confusing blob of code, but bear with me.  The logic is as follows.  Having already instantiated the neural network and it's parameters, you now want to define an update function that updates the networks's weights.  You define this function _inside_ the optimization function, then pass this function into the BaseUpdate function, which handles all the forward propagation and backwards propagation, using your function to define how to update neural network weights.  For those who know functional programming, this is a closure.

Why are optimization functions implemented this way?  In this fashion, you do not have to worry about backprob or the forward pass, you just care about how weights are updated.  Everything else is automatically calculated for you.  
An example is below.
```
def AdamOptimizer(X, Y, parameters, costs, activation_funcs, keep_probs,
        cost_func, reg_type, beta1, beta2, epsilon, **kwargs):

    # Pass in default configs
    configs = dict(default_configs)
    configs.update(kwargs)

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

    v_corrected = {}
    s_corrected = {}

    def update_parameters(parameters, grads, learning_rate):
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

        for l in range(L):
            v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta1**t)

            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)] * grads["dW" + str(l+1)]
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)] * grads["db" + str(l+1)]

            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1 - beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1 - beta2**t)

            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon))
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon))
        return parameters

    parameters, costs = BaseUpdate(X, Y, parameters, costs, activation_funcs, keep_probs, cost_func,
        reg_type, update_parameters, **configs)
    return parameters, costs
```

Here I give an example of defining other variables outside of update_parameters and using them inside the closure.
# TODO List and Features to Implement

- ~~Change db from scalar to vector of (layer size, 1)~~
- Activation Functions:
    - Softmax and derivative
    - ~~Tanh and derivative~~
    - ~~make passing in activation functions functional~~
    - ~~Pass activation functions into the cache to make it easily retrievable~~
    - ~~Remove dA from the calculations of derivatives.  Make custom implementations easier.~~
        - ~~Only need Z and derivative(bool) as parameters.  Move dA into backprop~~
- Optimization Functions:
    - Base Optimization
        - ~~Minibatch~~
        - Learning Rate Decay
    - ~~Gradient Descent~~
    - Classic Momentum
    - Nesterov Momentum
    - RMSProp
    - Adam
- Gradient Checks
- ~~Dropout layers~~
- Batch-Normalization
- Preprocessing Functions

# Bugs
- Dropout
    - Bug concerning numerical stability.  Sigmoid will output 1.0 only without dropout turned on.
    - Tried using a "stable" sigmoid, bug persisting.


