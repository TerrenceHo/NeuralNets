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


