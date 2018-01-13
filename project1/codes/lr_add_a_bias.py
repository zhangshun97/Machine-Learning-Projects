# adding a bias
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


class LeastSquareBias:

    def __init__(self):
        self.w = 0
        self.bias = 0

    def fit(self, train_set_x, train_set_y):
        """
        This is a least_square_bias regression model.
        :param train_set_x:pd.Series x
        :param train_set_y:pd.Series y as targets
        :return:a vector w including bias w_0
        """
        self.w = (sum(train_set_x*train_set_y))/(sum(train_set_x*train_set_x))

        self.bias = (sum(train_set_y)-self.w*sum(train_set_x))/len(train_set_x)

        average_train_error = sum((train_set_x * self.w + self.bias - train_set_y) ** 2) / len(train_set_x)
        print('average squared training error:', average_train_error)

    def test(self, test_x, test_y):
        """
        This is a test function for the regression above.
        :param test_x: pd.Series x
        :param test_y: pd.Series y as targets
        :return: a test error as a float
        """
        average_test_error = sum((test_x * self.w + self.bias - test_y) ** 2)/len(test_x)
        print('average squared testing error:', average_test_error)

    def show_plot(self, train_set_x, train_set_y):
        plt.scatter(train_set_x, train_set_y)
        plt.plot(train_set_x, (self.bias + self.w * train_set_x), color='red')
        plt.title('Adding a Bias')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


basic_data = load_data()
reg = LeastSquareBias()
reg.fit(basic_data['X'], basic_data['y'])
reg.test(basic_data['Xtest'], basic_data['Ytest'])
reg.show_plot(basic_data['X'], basic_data['y'])
