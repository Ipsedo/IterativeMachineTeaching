import numpy as np
import models.activations as act
import torch.nn as nn


class SingleLayer:
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.random.normal(0, 1. / np.sqrt(self.n_in + 1), size=(self.n_out, self.n_in + 1)) # +1 pour biais
        #self.biais = np.random.normal(0, 1. / np.sqrt(self.n_in), size=(self.n_out,))
        self.act = act.softmax
        self.eta = 1e-3

    def forward(self, X):
        # On rajoute des 1 aux Xs pour le biais
        return self.act(np.dot(np.concatenate((np.ones((X.shape[0], 1)), X), axis=1), self.weights.T))

    def compute_gradient(self, X, y):
        out = self.forward(X)
        one_hot_batch = self.to_one_hot_batch(y)
        derr = self.gradient_out(out, one_hot_batch)
        gbw = self.gradient(derr, X)
        return gbw

    def step(self, gbw):
        self.weights -= self.eta * gbw

    def to_one_hot_batch(self, labels):
        # On crée une matrice de labels.shape[0] lignes égales à [1,2,....,nblabel]
        labels = labels.reshape(-1, 1)
        tmp1 = np.tile(np.arange(self.n_out), (labels.shape[0], 1))  # [[1,2,....,nblabel],...,[1,2,....,nblabel]]
        res = np.zeros((labels.shape[0], self.n_out))  # resultat (batchSize, nbLabels) <- 0
        res[tmp1 == labels] = 1.0  # au bon index on affecte la valeur 1
        return res

    @staticmethod
    def gradient_out(out, one_hot_batch):
        return out - one_hot_batch

    @staticmethod
    def gradient(derror, X):
        # derror[batchSize, N] X[batchSize, M]
        # [batchSize, N] X[batchSize, M] -> dot([N, batchSize], X[batchSize, M]) -> [N,M]
        return np.dot(derror.T, np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)) / X.shape[0]


class LinearClassifier(nn.Module):
        def __init__(self, n_in):
            super(LinearClassifier, self).__init__()
            self.lin = nn.Linear(n_in, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            self.lin.weight.retain_grad()
            out = self.lin(x)
            return self.sig(out)
