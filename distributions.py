import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvnorm

from utils import *


class Distribution:
    def logp(self, args, data):
        raise NotImplementedError

    def dlogp(self, args, data):
        raise NotImplementedError


class UnivariateNormal(Distribution):
    def __init__(self):
        super(UnivariateNormal, self).__init__()

    def logp(self, args, data):
        """
            Calculates log probability of normal distribution
            Arguments:
                args = (mu, sigma)
                data = array of observed samples
            """
        if args[1] <= 0:
            return 0
        return np.sum(norm.logpdf(data, *args))

    def dlogp(self, args, data):
        """
            Calculates the gradient of the log probability of normal distribution
            Arguments:
                args = (mu, sigma)
                data = array of observed samples
            """
        mu, sig = args
        dmu = np.sum((data - mu) / (sig ** 2))
        dsig = np.sum((mu ** 2 - sig ** 2 - 2 * mu * data + data ** 2) / (sig ** 3))
        return np.array([dmu, dsig])


class IndependentMultivariateNormal(Distribution):
    def __init__(self, n_dims):
        super(IndependentMultivariateNormal, self).__init__()
        self.n_args = 2  *n_dims

    def logp(self, args, data):
        """
        Calculates log probability of normal distribution
        Arguments:
            args = (mu, sigma)
            data = array of observed samples
        """
        mus = args[:int(self.n_args / 2)]
        sigs = args[int(self.n_args / 2):]

        if (sigs <= 0).any():
            return 0
        return np.sum(mvnorm.logpdf(data, mus, np.diag(sigs)))

    def dlogp(self, args, data):
        """
        Calculates the gradient of the log probability of normal distribution
        Arguments:
            args = (mu, sigma)
            data = array of observed samples
        """
        mus = args[:int(self.n_args / 2)]
        sigs = args[int(self.n_args / 2):]

        dmu = np.sum((data - mus) / (sigs ** 2), axis=0)
        dsig = np.sum((mus ** 2 - sigs - 2 * mus * data + data ** 2) / (2 * sigs ** 2), axis=0)
        return np.concatenate([dmu, dsig])


class MultivariateNormal(Distribution):
    def __init__(self, n_dims):
        super(MultivariateNormal, self).__init__()
        self.ndims = n_dims
        self.n_args = int((self.ndims-1)*self.ndims/2 + self.ndims)

    # Define Model
    def logp(self, args, data):
        """
        Calculates log probability of normal distribution
        Arguments:
            args = (mu, sigma)
            data = array of observed samples
        """
        mu = np.zeros(self.ndims)
        Sig = np.linalg.inv(LT_to_mat(args, self.ndims))

        if (Sig <= 0).any():
            return 0
        return np.sum(mvnorm.logpdf(data, mu, Sig))

    def dlogp(self, args, data):
        """
        Calculates the gradient of the log probability of normal distribution
        Arguments:
            args = (mu, sigma)
            data = array of observed samples
        """
        Sig = np.linalg.inv(LT_to_mat(args, self.ndims))

        dTh_a = (2 * Sig - (Sig * np.identity(self.ndims)))
        dTh_b = np.sum([np.outer(x, x) for x in data])
        return mat_to_LT(len(data) * dTh_a - dTh_b)

