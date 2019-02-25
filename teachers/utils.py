import models.neural_network as neur_net
import torch as th
import torch.nn as nn


class BaseLinear(neur_net.LinearClassifier):
    def __init__(self, n_in):
        super(BaseLinear, self).__init__(n_in)
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


class BaseConv(neur_net.ConvModel):
    def __init__(self, eta):
        super(BaseConv, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.loss_fn.cuda()
        self.cuda()
        self.eta = eta
        self.optim = th.optim.SGD(self.parameters(), lr=self.eta)

    def update(self, X, y):
        self.train()
        self.optim.zero_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.step()
