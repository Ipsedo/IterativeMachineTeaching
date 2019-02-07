import teachers.omniscient_teacher as omni
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import sys
from sklearn.decomposition import PCA
import data.dataset_loader as dataset_loader


def mnist_data():
    nb_example = 5000
    nb_test = 1000

    mnistfile="./data/mnist.pkl.gz"
    train_set, valid_set, test_set = dataset_loader.load_mnist(mnistfile)

    X_train = np.asarray(train_set[0])
    Y_train = np.asarray(train_set[1])

    # Traitement PCA pour obtenir 24 composantes
    # pca = PCA(n_components = 24)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)

    class_1 = 1
    class_2 = 7
    f = (Y_train == class_1) | (Y_train == class_2)

    X = X_train[f]
    y = Y_train[f]
    y = np.where(y == class_1, 0, 1)

    X = X[:nb_example+nb_test]
    y = y[:nb_example+nb_test]

    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]

    print(X.shape)
    print(y.shape)

    batch_size = 5
    nb_batch = int(nb_example / batch_size)

    teacher = omni.OmniscientLinearClassifier(784)
    example = omni.OmniscientLinearClassifier(784)
    student = omni.OmniscientLinearClassifier(784)

    X_train = th.Tensor(X[:nb_example])
    y_train = th.Tensor(y[:nb_example]).view(-1)
    X_test = th.Tensor(X[nb_example:nb_example+nb_test])
    y_test = th.Tensor(y[nb_example:nb_example+nb_test])

    for e in range(30):
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            teacher.update(X_train[i_min:i_max], y_train[i_min:i_max])
        test = teacher(X_test)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1)).sum().item()
        print(nb_correct, "/", X_test.size(0))

    T = 300

    res_example = []

    for t in range(T):
        i = th.randint(0, nb_batch, size=(1,)).item()
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        data = X_train[i_min:i_max]
        label = y_train[i_min:i_max]
        example.update(data, label)
        test = example(X_test)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1)).sum().item()
        res_example.append(nb_correct / X_test.size(0))

    print("Base line trained\n")

    res_student = []

    for t in range(T):
        scores = []
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            data = X_train[i_min:i_max]
            label = y_train[i_min:i_max]
            eta = student.optim.param_groups[0]["lr"]
            s = (eta ** 2) * student.example_difficulty(data, label)
            s -= eta * 2 * student.example_usefulness(teacher.lin.weight, data, label)
            scores.append(s)

        i = np.argmin(scores)
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        x_t = X_train[i_min:i_max]
        y_t = y_train[i_min:i_max]

        student.update(x_t, y_t)

        test = student(X_test)
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1)).sum().item()
        res_student.append(nb_correct / X_test.size(0))

        sys.stdout.write("\r" + str(t) + "/" + str(T) + ", idx=" + str(i) + " " * 100)
        sys.stdout.flush()

    plt.plot(res_example, c='b', label="linear classifier")
    plt.plot(res_student, c='r', label="omniscient teacher & linear classifier")
    plt.title("MNIST Linear model (class : " + str(class_1) + ", " + str(class_2) + ")")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
