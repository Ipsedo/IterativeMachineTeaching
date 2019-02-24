import torch as th
import torch.nn as nn
import models.neural_network as neur_net


class SurrogateLinearClassifier(neur_net.LinearClassifier):
    def __init__(self, n_in):
        super(SurrogateLinearClassifier, self).__init__(n_in)
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

    def example_usefulness(self, loss_teacher, X, y):
        out = self(X)
        loss = self.loss_fn(out, y)
        return loss - loss_teacher
