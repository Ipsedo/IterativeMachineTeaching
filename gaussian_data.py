import teachers.omniscient_teacher as omni
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import sys


def gaussian_main():
    nb_data_per_class = 2000
    X1 = np.random.multivariate_normal([0, 0], np.identity(2), nb_data_per_class)
    y1 = np.ones((nb_data_per_class,))

    X2 = np.random.multivariate_normal([1.8, 1.8], np.identity(2), nb_data_per_class)
    y2 = np.zeros((nb_data_per_class,))

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    indices = np.indices((nb_data_per_class * 2,))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    teacher = omni.OmniscientLinearClassifier(2)
    example = omni.OmniscientLinearClassifier(2)
    student = omni.OmniscientLinearClassifier(2)

    X = th.Tensor(X).view(-1, 2)
    y = th.Tensor(y).view(-1)

    batch_size = 1
    nb_batch = int(2 * nb_data_per_class / batch_size)

    for e in range(30):
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            teacher.update(X[i_min:i_max], y[i_min:i_max])
        test = teacher(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        print(nb_correct, "/", 2 * nb_data_per_class)

    T = 2000

    res_example = []

    for t in range(T):
        i = th.randint(0, nb_batch, size=(1,)).item()
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        data = X[i_min:i_max]
        label = y[i_min:i_max]
        example.update(data, label)
        test = example(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        res_example.append(nb_correct / (2 * nb_data_per_class))

    print("Base line trained\n")

    res_student = []

    for t in range(T):
        scores = []
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            data = X[i_min:i_max]
            label = y[i_min:i_max]
            eta = student.optim.param_groups[0]["lr"]
            s = (eta ** 2) * student.example_difficulty(data, label)
            s -= eta * 2 * student.example_usefulness(teacher.lin.weight, data, label)
            scores.append(s)

        i = np.argmin(scores)
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        x_t = X[i_min:i_max]
        y_t = y[i_min:i_max]
        student.update(x_t, y_t)

        test = student(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        res_student.append(nb_correct / (2 * nb_data_per_class))

        sys.stdout.write("\r" + str(t) + "/" + str(T) + ", idx=" + str(i) + " " * 100)
        sys.stdout.flush()

    plt.plot(res_example, c='b')
    plt.plot(res_student, c='r')
    plt.show()
