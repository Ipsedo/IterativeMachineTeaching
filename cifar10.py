import data.dataset_loader as data_loader
import matplotlib.pyplot as plt
import teachers.omniscient_teacher as omni
import numpy as np
import torch as th
import sys
from tqdm import tqdm
import copy


def cifar10_main():
    data, labels = data_loader.load_cifar10_2()
    labels = labels.reshape(-1)

    nb_example = 9000
    nb_test = 500

    class_1 = 5
    class_2 = 9

    f = (labels == class_1) | (labels == class_2)
    data, labels = data[f], labels[f]

    data = data_loader.cifar10_proper_array(data)

    data, labels = data[:nb_example+nb_test], labels[:nb_example+nb_test]

    labels = np.where(labels == class_1, 0, 1)

    randomize = np.arange(data.shape[0])
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]

    print(data.shape)
    print(labels.shape)

    teacher = omni.OmniscientConvClassifier(2e-3)

    X = th.Tensor(data[:nb_example])
    y = th.Tensor(labels[:nb_example]).view(-1)
    X_test = th.Tensor(data[nb_example:nb_example+nb_test])
    y_test = th.Tensor(labels[nb_example:nb_example+nb_test]).view(-1)
    print(X_test.size())
    sys.stdout.flush()

    batch_size = 32
    nb_batch = int(nb_example / batch_size)

    accuracies = []
    for _ in tqdm(range(100)):
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            teacher.update(X[i_min:i_max].cuda(), y[i_min:i_max].cuda())
        teacher.eval()
        test = teacher(X_test.cuda()).cpu()
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1)).sum().item()
        accuracies.append(nb_correct / X_test.size(0))

    plt.plot(accuracies, c="b", label="Teacher (CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    example = copy.deepcopy(teacher)
    student = copy.deepcopy(teacher)

    th.nn.init.xavier_uniform_(example.lin.weight)
    example.optim = th.optim.SGD(example.lin.parameters(), lr=example.eta)

    th.nn.init.xavier_uniform_(student.lin.weight)
    student.optim = th.optim.SGD(student.lin.parameters(), lr=student.eta)

    T = 400

    nb_example = 200
    X = th.Tensor(data[:nb_example])
    y = th.Tensor(labels[:nb_example]).view(-1)
    X_test = th.Tensor(data[nb_example:nb_example + nb_test])
    y_test = th.Tensor(labels[nb_example:nb_example + nb_test]).view(-1)

    batch_size = 1
    nb_batch = int(nb_example / batch_size)

    res_example = []

    for t in range(T):
        i = th.randint(0, nb_batch, size=(1,)).item()
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        x_b = X[i_min:i_max].cuda()
        y_b = y[i_min:i_max].cuda()

        example.update(x_b, y_b)

        test = example(X_test.cuda()).cpu()
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1)).sum().item()
        res_example.append(nb_correct / X_test.size(0))

    print("Base line trained\n")

    res_student = []

    for t in range(T):
        min_score = sys.maxsize
        idx_min = 0
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size

            x_b = X[i_min:i_max].cuda()
            y_b = y[i_min:i_max].cuda()

            eta = student.optim.param_groups[-1]["lr"]

            s = (eta ** 2) * student.example_difficulty(x_b, y_b)
            s -= eta * 2 * student.example_usefulness(teacher.lin.weight, x_b, y_b)

            if min_score > s:
                min_score = s
                idx_min = i

        i = idx_min

        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        x_t = X[i_min:i_max].cuda()
        y_t = y[i_min:i_max].cuda()

        student.update(x_t, y_t)

        student.eval()
        test = student(X_test.cuda()).cpu()
        tmp = th.where(test > 0.5, th.ones(1), th.zeros(1))
        nb_correct = th.where(tmp.view(-1) == y_test, th.ones(1), th.zeros(1)).sum().item()
        res_student.append(nb_correct / X_test.size(0))

        sys.stdout.write("\r" + str(t) + "/" + str(T) + ", idx=" + str(i) + " " * 10)
        sys.stdout.flush()

    d = data_loader.cifar10_dictclass()
    plt.plot(res_example, c='b', label="CNN")
    plt.plot(res_student, c='r', label="omniscient teacher & CNN")
    plt.title("Cifar10 CNN (class : " + d[class_1] + ", " + d[class_2] + ")")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
