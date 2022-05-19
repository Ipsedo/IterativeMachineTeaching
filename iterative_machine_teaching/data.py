from typing import Tuple

import gzip
import pickle

import torch as th


def load_mnist(pickle_path: str) -> Tuple[th.Tensor, th.Tensor]:
    f = gzip.open(pickle_path, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin-1')
    f.close()

    train_x = th.from_numpy(train_set[0])
    train_y = th.from_numpy(train_set[1])

    valid_x = th.from_numpy(valid_set[0])
    valid_y = th.from_numpy(valid_set[1])

    test_x = th.from_numpy(test_set[0])
    test_y = th.from_numpy(test_set[1])

    x = th.cat([train_x, valid_x, test_x], dim=0)
    y = th.cat([train_y, valid_y, test_y], dim=0)

    rand_perm = th.randperm(x.size()[0])

    x = th.index_select(x, 0, rand_perm)
    y = th.index_select(y, 0, rand_perm)

    return x, y


def load_gaussian(nb_gaussian: int, dim: int, nb_data_per_class: int) -> Tuple[th.Tensor, th.Tensor]:
    x = []
    y = []

    for i in range(nb_gaussian):
        multivariate_dist = th.distributions.multivariate_normal.MultivariateNormal(
            th.rand(dim) * 2.0, th.eye(dim)
        )

        x.append(multivariate_dist.sample((nb_data_per_class,)))
        y.append(th.ones(nb_data_per_class, dtype=th.long) * i)

    x = th.cat(x, dim=0)
    y = th.cat(y, dim=0)

    rand_perm = th.randperm(x.size()[0])

    x = th.index_select(x, 0, rand_perm)
    y = th.index_select(y, 0, rand_perm)

    return x, y
