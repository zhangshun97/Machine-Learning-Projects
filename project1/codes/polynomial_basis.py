# regress with polynomial basis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """
    this function is about to load data from a txt file
    :return: a dataframe
    """
    with open('basicData.txt', 'r') as f:
        context = f.readlines()
        #print(context)
        column_names = context[0].split('\t')
        column_names[-1] = column_names[-1][:-1]

        data = [[] for i in range(4)]
        for i in range(1, len(context)):
            for j in range(4):
                if j == 3:
                    data[j].append(context[i].split('\t')[j][:-1])
                else:
                    data[j].append(context[i].split('\t')[j])

        data = pd.DataFrame(np.matrix(data, dtype='float').T, columns=column_names)
        #print(data)
    return data


def least_squares_basis(x, y, deg, xtest, ytest):
    """
    This is a polynomial regression.
    :param x: training set x (pd.Series)
    :param y: target set y (pd.Series)
    :param deg: the degree of the polynomial (int)
    :param xtest: test set x (pd.Series)
    :param ytest: test set y (pd.Series)
    :return: some errors
    """
    y = y.values.reshape(-1, 1)
    ytest = ytest.values.reshape(-1, 1)

    x_mat = []
    for d in range(deg+1):
        add_ = x ** d
        x_mat.append(add_)

    x_mat = np.matrix(x_mat).T

    coef = (x_mat.T * x_mat)**-1 * (x_mat.T * y)

    def get_value(x0):
        x_out = []
        for j in range(len(x0)):
            value = 0
            for i in range(deg+1):
                value += coef[i] * x0[j] ** i
            x_out.append(value)
        return pd.Series(x_out).values.reshape(-1, 1)

    # training error
    y_new = get_value(x)
    training_error.append(sum((y_new - y)**2)/len(x))
    print('training error:', training_error[-1])
    # testing error
    y_test_new = get_value(xtest)
    testing_error.append(sum((y_test_new - ytest) ** 2) / len(x))
    print('testing error:', testing_error[-1])
    print('-'*20)
    # plot
    plt.scatter(x, y)
    plt.plot(x, y_new, color='red')
    plt.show()


training_error = []
testing_error = []
basic_data = load_data()
for i in range(11):
    print("degree:", i)
    least_squares_basis(basic_data['X'], basic_data['y'],
                        i, basic_data['Xtest'], basic_data['Ytest'])

plt.plot(range(3, 11), training_error[3:], color='blue')
plt.plot(range(3, 11), testing_error[3:], color='red')
plt.legend(['training error', 'test error'])
plt.xlabel('degree')
plt.ylabel('average squared error')
plt.show()
