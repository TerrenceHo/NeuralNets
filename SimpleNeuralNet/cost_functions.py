def cross_entropy_loss(y_enc, output, weights, l2):
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
    for i in range(len(weights)):
        l2_term += (lf.l2/2.0) * (np.sum(np.square(weights[i][1:,:])))

    cost += l2_term
    return cost/y_enc.shape[0]

