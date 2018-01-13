# first we will standardize the import data
# then we will fit a ridge regression model

import numpy as np
import pandas as pd
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

        data_out = [[] for iii in range(len(data_names))]

        basic_data = f.readline()
        while basic_data:
            data_01 = basic_data.split('\t')
            data_01[-1] = data_01[-1][:-1]
            for i in range(len(data_01)):
                data_out[i].append(float(data_01[i]))
            basic_data = f.readline()

    data_out_attr = np.matrix(data_out[:-1]).T
    data_out_targ = np.array(data_out[-1]).T

    return pd.DataFrame(data_out_attr, columns=data_names[:-1]), pd.Series(data_out_targ, name=data_names[-1])


# a dataframe of attributes
attr_data = load_data()[0]
# a series of target
target_data = load_data()[1]

# random shuffle
slide_ = list(range(len(target_data)))
np.random.shuffle(slide_)
train_slide = slide_[:50]
test_slide = slide_[50:]

names = [i for i in attr_data]
train_attr = [[] for i in attr_data]
train_target = []

for a in train_slide:
    for i in range(len(names)):
        train_attr[i].append(attr_data[names[i]][a])
    train_target.append(target_data[a])
test_attr = [[] for i in attr_data]
test_target = []
for a in test_slide:
    for i in range(len(names)):
        test_attr[i].append(attr_data[names[i]][a])
    test_target.append(target_data[a])
train_attr = pd.DataFrame(train_attr).T
train_attr.columns = names
test_attr = pd.DataFrame(test_attr).T
test_attr.columns = names
train_target = np.array(train_target)
test_target = np.array(test_target)


# 标准化数据
def standardize_data(array):
    mean_train = np.mean(array)
    var_train = np.sqrt(np.var(array))
    array = (array - mean_train)/var_train
    return array


for i in range(len(names)):
    train_attr[names[i]] = standardize_data(train_attr[names[i]])

train_target = standardize_data(train_target)


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


theta_path = [[] for i in range(len(names))]
for d2 in np.linspace(0.05, 3000, 5000):
    theta_new = ridge(np.matrix(train_attr), np.array(train_target), d2)
    for i in range(len(names)):
        theta_path[i].append(theta_new[0, i])

plt.axes(xscale='log')
for i in range(len(names)):
    plt.plot(np.linspace(0.05, 3000, 5000), theta_path[i])

plt.title('Regularization Path')
plt.legend(names)
plt.show()
