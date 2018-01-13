import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array(range(10))
b = [[1,2], [2,3]]
c = pd.DataFrame(b, columns=['a', 'b'])
print(np.dot(a, a.T))
c['a'] = c['a']**2/4
d = np.matrix(b)
cc = pd.DataFrame(b)
print(d[0:2])
print('')
