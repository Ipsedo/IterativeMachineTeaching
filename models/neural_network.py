import torch.nn as nn


class LinearClassifier(nn.Module):
        def __init__(self, n_in):
            super(LinearClassifier, self).__init__()
            self.lin = nn.Linear(n_in, 1)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            out = self.lin(x)
            return self.sig(out)


# shape pour MNIST /!\
class ConvolutionClassifier(nn.Module):
    def __init__(self):
        super(ConvolutionClassifier, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(3, 3)),
                                 nn.MaxPool2d((3, 3)),
                                 nn.ReLU())

        self.lin = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.seq(x).view(-1, 8 * 8 * 8)
        out = self.lin(out)
        return self.sig(out)


# shape pour cifar10 /!\
class ConvModel(nn.Module):
    def __init__(self):
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
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
            one_data = True
        else:
            one_data = False

        out = self.seq(x).view(-1, self.linear1_dim)
        out = self.lin(out)
        out = self.sig(out)
        return out.squeeze(0) if one_data else out


class ConvModelMultiClass(nn.Module):
    def __init__(self, nb_class):
        img_size = 32
        super(ConvModelMultiClass, self).__init__()

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
        self.lin = nn.Linear(self.linear1_dim, nb_class)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
            one_data = True
        else:
            one_data = False
        out = self.seq(x).view(-1, self.linear1_dim)
        out = self.lin(out)
        return out.squeeze(0) if one_data else out
