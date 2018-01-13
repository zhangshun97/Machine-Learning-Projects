import numpy as np
from utils import sigmoid
import matplotlib.pyplot as plt

a = np.array(range(10)).reshape(-1, 1)
print(2 * a)
print(sigmoid(a).shape)
print(2*np.append(a, [11]))
print([1 if value > 5 else 0 for value in a])
print('aaa{0}a{1}'.format(a[1], a[2]))
b = np.matrix(a).reshape(5, 2)
c = [1,1,1,1,1]
print(np.c_[b, c])
pen_lambda = [0.001, 0.01, 0.1, 1]
valid_average_rate = [1, 2, 0, 2]
plt.xscale('log')
plt.plot(pen_lambda, valid_average_rate, color='green')
plt.title('average classification rate on validation set')
plt.ylabel('average classification rate')
plt.xlabel('lambda')
plt.show()
