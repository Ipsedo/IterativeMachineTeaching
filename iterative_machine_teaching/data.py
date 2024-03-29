# -*- coding: utf-8 -*-
import gzip
import pickle
from typing import Tuple

import torch as th


def load_mnist(pickle_path: str) -> Tuple[th.Tensor, th.Tensor]:
    f = gzip.open(pickle_path, "rb")
    train_set, valid_set, test_set = pickle.load(f, encoding="latin-1")
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


def load_gaussian(
    dim: int, nb_data_per_class: int
) -> Tuple[th.Tensor, th.Tensor]:
    x = []
    y = []

    size = th.Size([nb_data_per_class])

    # First center
    # rand_m = th.randn(dim, dim)
    # cov = rand_m.T @ rand_m
    cov = th.eye(dim)

    mean = th.ones(dim) * 0.5

    multivariate_dist = (
        th.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    )

    x.append(multivariate_dist.sample(size))
    y.append(th.ones(nb_data_per_class, dtype=th.long))

    # Second center
    # rand_m = th.randn(dim, dim)
    # cov = rand_m.T @ rand_m
    cov = th.eye(dim)

    mean = -th.ones(dim) * 0.5

    multivariate_dist = (
        th.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    )

    x.append(multivariate_dist.sample(size))
    y.append(th.zeros(nb_data_per_class, dtype=th.long))

    # concat all
    x_tensor = th.cat(x, dim=0)
    y_tensor = th.cat(y, dim=0)

    rand_perm = th.randperm(x_tensor.size()[0])

    x_tensor = th.index_select(x_tensor, 0, rand_perm)
    y_tensor = th.index_select(y_tensor, 0, rand_perm)

    return x_tensor, y_tensor
