# first we will standardize the import data
# then we will fit a ridge regression model
# here we apply ten-fold CV to find the best delta

import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """
    Load data and separate as attributes and target
    :return: a matrix and a array
    """
    with open('prostate.data.txt', 'r') as f:
        basic_data = f.readline()
        data_names = basic_data.split('\t')
        data_names[-1] = data_names[-1][:-1]

        data_out_attr = []
        data_out_targ = []

        basic_data = f.readline()
        while basic_data:
            data_01 = basic_data.split('\t')
            data_01[-1] = data_01[-1][:-1]
            for i in range(len(data_01)):
                data_01[i] = float(data_01[i])
            data_out_attr.append(data_01[:-1])
            data_out_targ.append(data_01[-1])
            basic_data = f.readline()

    return data_out_attr, data_out_targ


# a matrix of attributes
attr_data = load_data()[0]

# an array of target
target_data = load_data()[1]

# separate the data set into ten parts
slide_ = list(range(len(target_data)))
np.random.shuffle(slide_)
k_fold = 5  # the times of k-fold CV


def get_attr(slide):
    out = []
    for n in slide:
        out.append(attr_data[n])
    return np.matrix(out)


def get_target(slide):
    out = []
    for n in slide:
        out.append(target_data[n])
    return np.matrix(out).reshape(-1, 1)


# 标准化数据
def standardize_data(array, mean, std):
    array = (array - mean)/std
    return array


def ridge(x, y, d2):
    """
    This is a ridge regression function given delta^2 as d2
    :param x: attributes (matrix)
    :param y: target (array)
    :param d2: the ridge regression hypo-parameter
    :return:a 8-dim theta list
    """
    theta = (d2 * np.identity(len(x.T)) + x.T * x)**-1 * x.T
    theta = np.dot(theta, y)

    return theta


def test(x, y, theta_):
    y_star = np.dot(x, theta_)
    deviation = y - y_star
    mean_error = np.sqrt(np.dot(deviation.T, deviation))/np.sqrt(np.dot(y.T, y))
    return mean_error


test_mean_error = []
train_mean_error = []
delta2_set = np.logspace(-2, 3, 100)
for delta2 in delta2_set:
    test_error = []
    train_error = []
    # calculate error using 10-fold CV
    for i in range(k_fold):
        # create sets
        test_slide = slide_[20*i:20*(i+1)]
        train_slide = slide_[:20*i]+slide_[20*(i+1):]
        cv_train_attr = get_attr(train_slide)
        cv_test_attr = get_attr(test_slide)
        cv_train_target = get_target(train_slide)
        cv_test_target = get_target(test_slide)
        # standardize datas
        for j in range(len(cv_test_attr[0])):
            cv_train_attr[j] = standardize_data(cv_train_attr[j], np.mean(cv_train_attr[j]),
                                                   np.std(cv_train_attr[j]))
            cv_test_attr[j] = standardize_data(cv_test_attr[j], np.mean(cv_train_attr[j]),
                                                  np.std(cv_train_attr[j]))
        cv_train_target = standardize_data(cv_train_target, np.mean(cv_train_target),
                                           np.std(cv_train_target))
        cv_test_target = standardize_data(cv_test_target, np.mean(cv_test_target),
                                          np.std(cv_test_target))
        # fit
        theta = ridge(cv_train_attr, cv_train_target, delta2)
        # test
        train_error.append(test(cv_train_attr, cv_train_target, theta))
        test_error.append(test(cv_test_attr, cv_test_target, theta))
    train_mean_error.append(np.mean(train_error))
    test_mean_error.append(np.mean(test_error))

plt.axes(xscale='log')
plt.plot(delta2_set, train_mean_error)
plt.plot(delta2_set, test_mean_error)
plt.legend(['train error', 'test error'])
plt.show()
