import sklearn.linear_model as lin
import models.neural_network as neur_net
import torch.nn as nn
import torch as th
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


class OmniscientSingleLayer(neur_net.SingleLayer):
    def __init__(self, n_in, n_out):
        super(OmniscientSingleLayer, self).__init__(n_in, n_out)

    def example_difficulty(self, X, y):
        gwb = self.compute_gradient(X, y)
        return np.linalg.norm(gwb) ** 2

    def example_use_fullness(self, w_star, X, y):
        gwb = self.compute_gradient(X, y)
        diff = self.weights - w_star
        return np.dot(diff.T, gwb)


class OmniscientLinearClassifier(neur_net.LinearClassifier):
    def __init__(self, n_in):
        super(OmniscientLinearClassifier, self).__init__(n_in)
        self.loss_fn = nn.MSELoss()
        self.eta = 1e-3
        self.optim = th.optim.SGD(self.parameters(), lr=self.eta)

    def update(self, X, y):
        self.train()
        self.optim.zero_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.step()

    def example_difficulty(self, X, y):
        self.train()
        self.optim.zero_grad()
        self.lin.weight.retain_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        res = self.lin.weight.grad
        return (th.norm(res) ** 2).item()

    def example_usefulness(self, w_star, X, y):
        self.train()
        self.optim.zero_grad()
        self.lin.weight.retain_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        res = self.lin.weight.grad
        diff = self.lin.weight - w_star
        return th.dot(diff.view(-1), res.view(-1)).item()


class OmniscientConvClassifier(neur_net.ConvModel):
    def __init__(self):
        super(OmniscientConvClassifier, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.loss_fn.cuda()
        self.cuda()
        self.eta = 5e-2
        self.optim = th.optim.SGD(self.parameters(), lr=self.eta)

    def update(self, X, y):
        self.train()
        self.optim.zero_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.step()

    def example_difficulty(self, X, y):
        self.train()
        self.optim.zero_grad()
        self.lin.weight.retain_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        res = self.lin.weight.grad
        return (th.norm(res) ** 2).item()

    def example_usefulness(self, w_star, X, y):
        self.train()
        self.optim.zero_grad()
        self.lin.weight.retain_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        res = self.lin.weight.grad
        diff = self.lin.weight - w_star
        return th.dot(diff.view(-1), res.view(-1)).item()
