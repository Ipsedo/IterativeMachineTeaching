# -*- coding: utf-8 -*-
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from ..data import load_mnist
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


def mnist_main(teacher_type):
    dim = 784

    nb_example = 5000
    nb_test = 1000

    # Chargement des données
    mnistfile = "./data/mnist.pkl.gz"
    train_set, valid_set, test_set = load_mnist(mnistfile)

    X_train = np.asarray(train_set[0])
    Y_train = np.asarray(train_set[1])

    class_1 = 1
    class_2 = 7

    # Séléction des deux classes souhaitées
    f = (Y_train == class_1) | (Y_train == class_2)
    X = X_train[f]
    y = Y_train[f]

    # Renomage des labels
    y = np.where(y == class_1, 0, 1)

    # On prend le bon nombre de données
    X = X[: nb_example + nb_test]
    y = y[: nb_example + nb_test]

    # Shuffle des données
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]

    print(X.shape)
    print(y.shape)

    batch_size = 5
    nb_batch = int(nb_example / batch_size)

    # Création de l'exemple
    example = BaseLinear(dim)

    # Séléction du teacher et du student
    if teacher_type == "omni":
        teacher = OmniscientLinearTeacher(dim)
        student = OmniscientLinearStudent(dim)
        teacher_name = "omniscient teacher"
    elif teacher_type == "surro_same":
        teacher = SurrogateLinearTeacher(dim)
        student = SurrogateLinearStudent(dim)
        teacher_name = "surrogate teacher (same feature space)"
    elif teacher_type == "surro_diff":
        teacher = SurrogateDiffLinearTeacher(dim, 24, normal_dist=True)
        student = SurrogateLinearStudent(dim)
        teacher_name = "surrogate teacher (different feature space)"
    elif teacher_type == "immi_diff":
        fst_x = th.Tensor(X[th.randint(0, X.shape[0], (1,)).item()])
        teacher = ImitationDiffLinearTeacher(dim, 24, fst_x, normal_dist=True)
        student = BaseLinear(dim)
        teacher_name = "immitation teacher (different feature space)"
    elif teacher_type == "immi_same":
        fst_x = th.Tensor(X[th.randint(0, X.shape[0], (1,)).item()])
        teacher = ImitationLinearTeacher(dim, fst_x)
        student = BaseLinear(dim)
        teacher_name = "immitation teacher (same feature space)"
    else:
        print("Unrecognized teacher !")
        sys.exit()

    # Passage des données vers pytorch
    X_train = th.Tensor(X[:nb_example])
    y_train = th.Tensor(y[:nb_example]).view(-1)
    X_test = th.Tensor(X[nb_example : nb_example + nb_test])
    y_test = th.Tensor(y[nb_example : nb_example + nb_test])

    # Entrainement du teacher
    for e in range(30):
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            teacher.update(X_train[i_min:i_max], y_train[i_min:i_max])
        test = teacher(X_test)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = (
            th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1))
            .sum()
            .item()
        )
        print(nb_correct, "/", X_test.size(0))

    T = 300

    res_example = []

    # Entrainement de l'exemple
    for t in range(T):
        i = th.randint(0, nb_batch, size=(1,)).item()
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        data = X_train[i_min:i_max]
        label = y_train[i_min:i_max]
        example.update(data, label)
        test = example(X_test)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = (
            th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1))
            .sum()
            .item()
        )
        res_example.append(nb_correct / X_test.size(0))

    print("Base line trained\n")

    res_student = []

    # Entrainement du student avec le teacher
    for t in range(T):
        i = teacher.select_example(student, X_train, y_train, batch_size)
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        x_t = X_train[i_min:i_max]
        y_t = y_train[i_min:i_max]

        student.update(x_t, y_t)

        test = student(X_test)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = (
            th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1))
            .sum()
            .item()
        )
        res_student.append(nb_correct / X_test.size(0))

        sys.stdout.write(
            "\r" + str(t) + "/" + str(T) + ", idx=" + str(i) + " " * 100
        )
        sys.stdout.flush()

    plt.plot(res_example, c="b", label="linear classifier")
    plt.plot(res_student, c="r", label="%s & linear classifier" % teacher_name)
    plt.title(
        "MNIST Linear model (class : "
        + str(class_1)
        + ", "
        + str(class_2)
        + ")"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
