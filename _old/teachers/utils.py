from ..models import LinearClassifier, ConvModel
import torch as th
import torch.nn as nn


class BaseLinear(LinearClassifier):
    """
    Modèle linéaire de base.
    Contient le modèle (lui-même), la fonction de perte et l'optimiseur
    """
    def __init__(self, n_in):
        super(BaseLinear, self).__init__(n_in)
        self.loss_fn = nn.MSELoss()
        self.eta = 1e-3
        self.optim = th.optim.SGD(self.parameters(), lr=self.eta)

    def update(self, X, y):
        """
        Méthode d'apprentissage
        :param X: La données / le batch de données
        :param y: Le label / le batch de labels
        :return: Rien (procedure)
        """
        self.train()
        self.optim.zero_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.step()


class BaseConv(ConvModel):
    """
    Modèle à cnvolution de base.
    Contient le modèle (lui-même), la fonction de perte, et l'optimiseur.
    """
    def __init__(self, eta):
        super(BaseConv, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.loss_fn.cuda()
        self.cuda()
        self.eta = eta
        self.optim = th.optim.SGD(self.parameters(), lr=self.eta)

    def update(self, X, y):
        """
        Méthode d'apprentissage
        :param X: Les données d'apprentissage
        :param y: Les labels
        :return: Rien (procedure)
        """
        self.train()
        self.optim.zero_grad()
        out = self(X)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.step()
