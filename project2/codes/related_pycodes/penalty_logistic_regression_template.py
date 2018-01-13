import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *


def run_logistic_regression(weight_regularization):
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = np.shape(train_inputs)

    # TODO: Set hyperparameters
    hyperparameters = {
        'learning_rate': 0.01,
        'weight_regularization': weight_regularization,
        'num_iterations': 1000
    }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.rand(M + 1, 1) * 0.1

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # TODO: you may need to modify this loop to create plots, etc.
    train_frac = []
    valid_frac = []
    ce_train = []
    ce_valid = []

    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
        stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
        '''
        print(stat_msg.format(t+1,
                              float(f / N),
                              float(cross_entropy_train),
                              float(frac_correct_train*100),
                              float(cross_entropy_valid),
                              float(frac_correct_valid*100)))
        '''
        train_frac.append(frac_correct_train * 100)
        valid_frac.append(frac_correct_valid * 100)
        ce_train.append(cross_entropy_train)
        ce_valid.append(cross_entropy_valid)
    '''
    plt.plot(list(range(hyperparameters['num_iterations'])), train_frac, color='blue')
    plt.plot(list(range(hyperparameters['num_iterations'])), valid_frac, color='green')
    plt.title('penalty logistic regression (lambda = {})'.format(hyperparameters['weight_regularization']))
    plt.xlabel('the number of iteration')
    plt.ylabel('predict correct rate')
    plt.legend(['train({}%)'.format(train_frac[-1]), 'valid({}%)'.format(valid_frac[-1])])
    plt.show()
    '''
    return train_frac[-1], valid_frac[-1], ce_train[-1], ce_valid[-1]


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)

    diff = check_grad(logistic_pen,  # function to check
                      weights,
                      0.001,  # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == '__main__':
    train_average_rate = []
    valid_average_rate = []
    train_average_ce = []
    valid_average_ce = []
    pen_lambda = [0.001, 0.01, 0.1, 1]
    for _lambda in pen_lambda:
        train_rate = []
        valid_rate = []
        train_ce = []
        valid_ce = []
        for ii in range(10):
            rt, rv, ct, cv = run_logistic_regression(_lambda)
            train_rate.append(rt)
            train_ce.append(ct)
            valid_rate.append(rv)
            valid_ce.append(cv)
        train_average_rate.append(np.mean(train_rate))
        train_average_ce.append(np.mean(train_ce))
        valid_average_rate.append(np.mean(valid_rate))
        valid_average_ce.append(np.mean(valid_ce))

    plt.xscale('log')
    plt.plot(pen_lambda, train_average_rate, color='blue')
    plt.plot(pen_lambda, valid_average_rate, color='green')
    plt.title('average classification rate')
    plt.ylabel('classification rate')
    plt.xlabel('lambda')
    plt.legend(['train', 'valid'])
    plt.show()

    plt.xscale('log')
    plt.plot(pen_lambda, train_average_ce, color='blue')
    plt.plot(pen_lambda, valid_average_ce, color='green')
    plt.title('average cross entropy')
    plt.ylabel('cross entropy')
    plt.xlabel('lambda')
    plt.legend(['train', 'valid'])
    plt.show()
