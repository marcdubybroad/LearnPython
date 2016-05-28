

# import
import numpy as np

# setup learning data
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# get the naive bauyes library
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x, y)

# print result
print(clf.predict([[-0.8, -1]]))
