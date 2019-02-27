import torch.nn as nn


class LinearClassifier(nn.Module):
        def __init__(self, n_in):
            """
            Constructeur classifieur linéaire simple
            Classification binaire (une seule sortie)
            :param n_in: nombre de features
            """
            super(LinearClassifier, self).__init__()
            self.lin = nn.Linear(n_in, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            """
            Méthode forward du modèle
            :param x: la donnée de size = (batch_size, nb_features) ou (nb_features)
            :return: la sortie du réseau à simple couche
            """
            out = self.lin(x)
            return self.sig(out)


# shape pour cifar10 /!\
class ConvModel(nn.Module):
    def __init__(self):
        """
        Constructeur modèle convolutionnel.
        Dimension réglées pour CIFAR-10
        """
        img_size = 32
        super(ConvModel, self).__init__()

        # conv1 : img 3 * 32 * 32 -> img 20 * 28 * 28
        # maxpool1 : img 20 * 28 * 28 -> img 20 * 14 * 14
        # conv2 : img 20 * 14 * 14 -> img 50 * 10 * 10
        # maxpool2 : img 50 * 10 * 10 -> 50 * 5 * 5

        self.seq = nn.Sequential(nn.Conv2d(3, 20, (5,5)),
                                 nn.MaxPool2d((2, 2), stride=(2, 2)),
                                 nn.ReLU(),
                                 nn.Conv2d(20, 50, (5, 5)),
                                 nn.MaxPool2d((2, 2), stride=(2, 2)),
                                 nn.ReLU())

        self.linear1_dim = int((((img_size - 4) / 2 - 4) / 2) ** 2 * 50)
        self.lin = nn.Linear(self.linear1_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Méthode forward
        :param x: image de size = (nb_batch, 3, 32, 32)
        :return: La sortie du réseau de size = (nb_batch, 1)
        """
        # pour rajouter une dimension pour le batch
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
            one_data = True
        else:
            one_data = False

        out = self.seq(x).view(-1, self.linear1_dim)
        out = self.lin(out)
        out = self.sig(out)
        return out.squeeze(0) if one_data else out
