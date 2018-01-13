""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    y1 = np.dot(data, weights[:-1]) + weights[-1]
    y = sigmoid(y1)

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function

    frac_correct = np.mean([1 if np.abs(targets[i] - y[i]) < 0.5 else 0 for i in range(len(y))])

    ce = 0
    for i in range(len(y)):
        ce -= np.log(y[i]) if targets[i] > 0.5 else np.log(1-y[i])

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    N, M = np.shape(data)

    y = logistic_predict(weights, data)

    data = np.c_[data, np.ones(N)]

    f = 0
    for i in range(len(y)):
        f -= np.log(y[i]) if targets[i] > 0.5 else np.log(1 - y[i])

    df = np.dot(data.T, sigmoid(np.dot(data, weights)) - targets)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function
    pen_lambda = hyperparameters['weight_regularization']
    N, M = np.shape(data)
    y = logistic_predict(weights, data)

    data = np.c_[data, np.ones(N)]

    f = 0
    for i in range(len(y)):
        f -= np.log(y[i]) if targets[i] > 0.5 else np.log(1 - y[i])

    f += pen_lambda/2 * np.dot(weights[:-1].T, weights[:-1])[0]

    df = np.dot(data.T, sigmoid(np.dot(data, weights)) - targets)
    pen_weights = pen_lambda * weights
    pen_weights[-1] = 0
    df += pen_weights

    return f, df, y
