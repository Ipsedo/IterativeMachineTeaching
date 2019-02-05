import sklearn.linear_model as lin
import models.neural_network as nn
import numpy as np


class OmniscientLinearRegression(lin.LinearRegression):
    def __init__(self):
        super(OmniscientLinearRegression, self).__init__()

    def example_difficulty(self, X, y):
        return (np.dot(self.coef_, X) - y) ** 2

    def example_usefulness(self):
        return None


class OmniscientSGDClassifier(lin.SGDClassifier):
    def __init__(self):
        super(OmniscientSGDClassifier, self).__init__()

    def example_difficulty(self, X, y):
        pass


class OmniscientSingleLayer(nn.SingleLayer):
    def __init__(self, n_in, n_out):
        super(OmniscientSingleLayer, self).__init__(n_in, n_out)

    def example_difficulty(self, X, y):
        gwb = self.compute_gradient(X, y)
        return np.linalg.norm(gwb) ** 2

    def example_use_fullness(self, w_star, X, y):
        gwb = self.compute_gradient(X, y)
        diff = self.weights - w_star
        return np.dot(diff.T, gwb)
