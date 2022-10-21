from data import data
from scipy.stats import multivariate_normal
import numpy as np


def estimate_params(xs: np.array):
    return np.mean(xs, axis=0), np.cov(xs, rowvar=False)


def split_data(xs: np.array, leave_out: int = False):
    if leave_out:
        return np.delete(xs, leave_out), xs[leave_out]
    else:
        return xs[0::2], xs[1::2]


X = np.array(data)
labels = np.unique(X[:, -1])
train_data = [None for _ in labels]
test_data = [None for _ in labels]

for i in range(labels.size):
    train_data[i], test_data[i] = split_data(X[X[:, -1] == labels[i]])

train_data = np.vstack(tuple(train_data))
test_data = np.vstack(tuple(test_data))