import teachers.omniscient_teacher as omni
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import sys
from sklearn.decomposition import PCA
import data.dataset_loader as dataset_loader

def mnist_data():

    mnistfile="./data/mnist.pkl.gz"
    train_set, valid_set, test_set = dataset_loader.load_mnist(mnistfile)

    X_train = np.asarray(train_set[0])
    Y_train = np.asarray(train_set[1])

    # Traitement PCA pour obtenir 24 composantes
    # pca = PCA(n_components = 24)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)

    X = X_train[Y_train <= 1]
    y = Y_train[Y_train <= 1]

    X = X[:2000]
    y = y[:2000]

    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]

    print(X.shape)
    print(y.shape)

    teacher = omni.OmniscientLinearClassifier(784)
    example = omni.OmniscientLinearClassifier(784)
    student = omni.OmniscientLinearClassifier(784)

    X = th.Tensor(X)
    y = th.Tensor(y).view(-1)

    for e in range(30):
        for i in range(200):
            i_min = i * 10
            i_max = (i + 1) * 10
            teacher.update(X[i_min:i_max], y[i_min:i_max])
        test = teacher(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        print(nb_correct, "/", X.size(0))

    T = 200

    res_example = []

    for t in range(T):
        i = th.randint(0, X.size(0), size=(1,)).item()
        data = X[i]
        label = y[i]
        example.update(data, label)
        test = example(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        res_example.append(nb_correct / X.size(0))

    print("Base line trained\n")

    res_student = []

    for t in range(T):
        scores = []
        for data, label in zip(X, y):
            eta = student.optim.param_groups[0]["lr"]
            s = (eta ** 2) * student.example_difficulty(data, label)
            s -= eta * 2 * student.example_usefulness(teacher.lin.weight, data, label)
            scores.append(s)

        i = np.argmin(scores)
        x_t = X[i]
        y_t = y[i]
        student.update(x_t, y_t)

        test = student(X)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y, th.ones(1), th.zeros(1)).sum().item()
        res_student.append(nb_correct / X.size(0))

        sys.stdout.write("\r" + str(t) + "/" + str(T) + ", idx=" + str(i) + " " * 100)
        sys.stdout.flush()

    plt.plot(res_example, c='b')
    plt.plot(res_student, c='r')
    plt.show()