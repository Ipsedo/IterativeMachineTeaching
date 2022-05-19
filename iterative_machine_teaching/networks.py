from abc import ABC, abstractmethod

import torch as th
import torch.nn as nn


class Classifier(ABC, nn.Module):
    @property
    @abstractmethod
    def linear(self) -> nn.Linear:
        pass


class LinearClassifier(Classifier):
    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        super(LinearClassifier, self).__init__()

        self.__lin = nn.Linear(input_dim, output_dim)

    @property
    def linear(self) -> nn.Linear:
        return self.__lin

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__lin(x)


class Cifar10Classifier(Classifier):
    def __init__(self, output_size: int):
        super(Cifar10Classifier, self).__init__()

        self.__seq = nn.Sequential(
            nn.Conv2d(
                3, 6,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 16 * 16

            nn.Conv2d(
                6, 12,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 8 * 8

            nn.Conv2d(
                12, 18,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 4 * 4

            nn.Conv2d(
                18, 24,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 2 * 2

            nn.Flatten(
                1, -1
            ),

            nn.Linear(
                2 * 2 * 24,
                output_size
            )
        )

    @property
    def linear(self) -> nn.Linear:
        return self.__seq[-1]

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__seq(x)


class ModelWrapper(object):
    def __init__(self, clf: Classifier, learning_rate: float):
        super(ModelWrapper, self).__init__()

        self._clf = clf

        self.__loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.__optim = th.optim.SGD(
            self._clf.parameters(),
            lr=learning_rate
        )

    def train(self, x: th.Tensor, y: th.Tensor) -> float:
        out = self._clf(x)
        loss = self.__loss_fn(out, y)

        self.__optim.zero_grad()
        loss.backward()
        self.__optim.step()

        return loss.item()

    def predict(self, x: th.Tensor) -> th.Tensor:
        return self._clf(x)

    def get_eta(self) -> float:
        return self.__optim.param_groups[0]["lr"]
