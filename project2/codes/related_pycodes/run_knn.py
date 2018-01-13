from l2_distance import l2_distance
import matplotlib.pyplot as plt
from utils import *


def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


# load data
train_data_ = np.load('mnist_train.npz')
test_data_ = np.load('mnist_test.npz')
valid_data_ = np.load('mnist_valid.npz')
train_data_inputs = train_data_['train_inputs']
train_data_labels = train_data_['train_targets']
test_data_inputs = test_data_['test_inputs']
valid_data_inputs = valid_data_['valid_inputs']

test_data_labels = test_data_['test_targets']
valid_data_labels = valid_data_['valid_targets']

train_inputs, train_targets = load_train_small()


def get_the_classification_rate(predict_labels, true_labels):
    c = 0
    n = len(predict_labels)
    for i in range(n):
        if predict_labels[i] == true_labels[i]:
            c += 1
    return c/n


test_classification_rates = []
valid_classification_rates = []
for k in [1, 3, 5, 7, 9]:
    test_predict_labels = run_knn(k, train_inputs, train_targets, test_data_inputs)
    test_classification_rates.append(get_the_classification_rate(test_predict_labels, test_data_labels))
    valid_predict_labels = run_knn(k, train_inputs, train_targets, valid_data_inputs)
    valid_classification_rates.append(get_the_classification_rate(valid_predict_labels, valid_data_labels))

plt.plot([1, 3, 5, 7, 9], test_classification_rates, color='red')
plt.plot([1, 3, 5, 7, 9], valid_classification_rates, color='green')
plt.title('k-Nearest Neighbours')
plt.xlabel('k')
plt.ylabel('classification rate')
plt.legend(['test set', 'valid set'])
plt.show()

print(test_classification_rates)
