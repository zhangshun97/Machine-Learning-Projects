import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = np.shape(train_inputs)

    # TODO: Set hyperparameters
    hyperparameters = {
        'learning_rate': 0.01,
        'weight_regularization': 0.1,
        'num_iterations': 1000
    }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.rand(M + 1, 1) * 0.1

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # TODO: you may need to modify this loop to create plots, etc.
    train_error = []
    valid_error = []
    test_error = []
    ce_train = []
    ce_valid = []
    ce_test = []

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

        # Make a prediction on the valid_inputs.
        predictions_test = logistic_predict(weights, test_inputs)

        # Evaluate the prediction.
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)

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
        train_error.append((1 - frac_correct_train) * 100)
        valid_error.append((1 - frac_correct_valid) * 100)
        test_error.append((1 - frac_correct_test) * 100)
        ce_train.append(cross_entropy_train)
        ce_valid.append(cross_entropy_valid)
        ce_test.append(cross_entropy_test)

    plt.plot(list(range(hyperparameters['num_iterations'])), train_error, color='blue')
    plt.plot(list(range(hyperparameters['num_iterations'])), valid_error, color='green')
    plt.plot(list(range(hyperparameters['num_iterations'])), test_error, color='red')
    plt.title('logistic regression(lambda = {})'.format(hyperparameters['weight_regularization']))
    plt.xlabel('the number of iteration')
    plt.ylabel('classification error')
    plt.legend(['train(%1.2f)' % train_error[-1],
                'valid(%1.2f)' % valid_error[-1],
                'test(%1.2f)' % test_error[-1]])
    plt.show()

    print(ce_train[-1])
    print(ce_valid[-1])
    print(ce_test[-1])


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
    run_logistic_regression()
