# -*- coding: utf-8 -*-
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from ..teachers import (
    BaseLinear,
    ImitationDiffLinearTeacher,
    ImitationLinearTeacher,
    OmniscientLinearStudent,
    OmniscientLinearTeacher,
    SurrogateDiffLinearTeacher,
    SurrogateLinearStudent,
    SurrogateLinearTeacher,
)


def init_data(dim, nb_data_per_class):
    """
    Création des données gaussien
    :param dim: la dimension des données
    :param nb_data_per_class: le nombre d'exemple par classe
    :return: un tuple (données, labels)
    """
    X1 = np.random.multivariate_normal(
        [0.5] * dim, np.identity(dim), nb_data_per_class
    )
    y1 = np.ones((nb_data_per_class,))

    X2 = np.random.multivariate_normal(
        [-0.5] * dim, np.identity(dim), nb_data_per_class
    )
    y2 = np.zeros((nb_data_per_class,))

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    indices = np.indices((nb_data_per_class * 2,))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]
    return X, y


def gaussian_main(teacher_type):
    dim = 10
    dim__diff = 7
    nb_data_per_class = 400

    X, y = init_data(dim, nb_data_per_class)

    # définition de l'exemple
    example = BaseLinear(dim)

    # Selection du student et du teacher
    if teacher_type == "omni":
        teacher = OmniscientLinearTeacher(dim)
        student = OmniscientLinearStudent(dim)
        teacher_name = "omniscient teacher"
    elif teacher_type == "surro_same":
        teacher = SurrogateLinearTeacher(dim)
        student = SurrogateLinearStudent(dim)
        teacher_name = "surrogate teacher (same feature space)"
    elif teacher_type == "surro_diff":
        teacher = SurrogateDiffLinearTeacher(dim, dim__diff, normal_dist=False)
        student = SurrogateLinearStudent(dim)
        teacher_name = "surrogate teacher (different feature space)"
    elif teacher_type == "immi_same":
        fst_x = th.Tensor(X[th.randint(0, X.shape[0], (1,)).item()])
        teacher = ImitationLinearTeacher(dim, fst_x)
        student = BaseLinear(dim)
        teacher_name = "immitation teacher (same feature space)"
    elif teacher_type == "immi_diff":
        fst_x = th.Tensor(X[th.randint(0, X.shape[0], (1,)).item()])
        teacher = ImitationDiffLinearTeacher(
            dim, dim__diff, fst_x, normal_dist=False
        )
        student = BaseLinear(dim)
        teacher_name = "immitation teacher (different feature space)"
    else:
        print("Unrecognized teacher !")
        sys.exit()

    X = th.Tensor(X).view(-1, dim)
    y = th.Tensor(y).view(-1)

    batch_size = 1
    nb_batch = int(2 * nb_data_per_class / batch_size)

    # entrainement du teacher
    for e in range(30):
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            teacher.update(X[i_min:i_max], y[i_min:i_max])
        test = teacher(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = (
            th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        )
        print(nb_correct, "/", 2 * nb_data_per_class)

    T = 500

    res_example = []

    # Entrainement de l'exemple
    for t in range(T):
        i = th.randint(0, nb_batch, size=(1,)).item()
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        data = X[i_min:i_max]
        label = y[i_min:i_max]
        example.update(data, label)
        test = example(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = (
            th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        )
        res_example.append(nb_correct / (2 * nb_data_per_class))

    print("Base line trained\n")

    res_student = []

    # Entrainement du student avec le teacher
    for t in range(T):
        i = teacher.select_example(student, X, y, batch_size)
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        x_t = X[i_min:i_max]
        y_t = y[i_min:i_max]
        student.update(x_t, y_t)

        test = student(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = (
            th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        )
        res_student.append(nb_correct / (2 * nb_data_per_class))

        sys.stdout.write(
            "\r" + str(t) + "/" + str(T) + ", idx=" + str(i) + " " * 100
        )
        sys.stdout.flush()

    plt.plot(res_example, c="b", label="linear classifier")
    plt.plot(res_student, c="r", label="%s & linear classifier" % teacher_name)
    plt.title("Gaussian data")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
