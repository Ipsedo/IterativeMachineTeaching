import numpy as np
import models.activations as act


class SingleLayer:
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.random.normal(0, 1. / np.sqrt(self.n_in), size=(self.n_out, self.n_in))
        self.biais = np.random.normal(0, 1. / np.sqrt(self.n_in), size=(self.n_out,))
        self.act = act.softmax

    def forward(self, x):
        return self.act(np.dot(x, self.weights) + self.biais)

    def compute_gradient(self, loss_values):
        #TODO pas plutôt besoin d'un set gradient avec omnicient teacher ?
        return 0

    def loss(self, out, y):
        #TODO loss
        return 0

    def step(self):
        #TODO apply the gradient on the single layer
        pass

