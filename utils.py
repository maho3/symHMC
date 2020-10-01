import numpy as np


def mat_to_LT(X):
    y = np.tril(X).flatten()
    return y[y != 0]


def LT_to_mat(y, ndims):
    X = np.zeros((ndims, ndims))
    X[np.tril_indices(ndims)] = y
    X += X.T - np.diag(np.diagonal(X))
    return X