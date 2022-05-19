import matplotlib.pyplot as plt
from ..teachers import (
    ImmitationConvTeacher,
    OmniscientConvTeacher,
    SurrogateConvTeacher,

    SurrogateConvStudent,
    OmniscientConvStudent,
    BaseConv
)
from ..data import load_cifar10_2, cifar10_proper_array
import numpy as np
import torch as th
import sys
from tqdm import tqdm
import copy


def cifar10_main(teacher_type):

    # Chargelent des données
    data, labels = load_cifar10_2()
    labels = labels.reshape(-1)

    nb_example = 9000
    nb_test = 500

    class_1 = 5
    class_2 = 9

    # Récupération des deux classes souhaitées
    f = (labels == class_1) | (labels == class_2)
    data, labels = data[f], labels[f]

    # Mise en bonne dimension des données
    data = cifar10_proper_array(data)

    # séléction du bon nombre de données
    data, labels = data[:nb_example + nb_test], labels[:nb_example + nb_test]

    # Changement de label et 0 | 1
    labels = np.where(labels == class_1, 0, 1)

    # Shuffle des données
    randomize = np.arange(data.shape[0])
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]

    print(data.shape)
    print(labels.shape)

    eta = 2e-3

    # Séléction du teacher
    if teacher_type == "omni":
        teacher = OmniscientConvTeacher(eta)
        teacher_name = "omniscient teacher"
    elif teacher_type == "surro_same":
        teacher = SurrogateConvTeacher(eta)
        teacher_name = "surrogate teacher (same feature space)"
    elif teacher_type == "surro_diff":
        print("Unsuported teacher for CIFAR-10 (%s) !" % teacher_type)
        sys.exit()
    elif teacher_type == "immi_same":
        fst_x = th.Tensor(data[th.randint(0, data.shape[0], (1,)).item()]).unsqueeze(0).cuda()
        teacher = ImmitationConvTeacher(eta, fst_x)
        teacher_name = "immitation teacher (same feature space)"
    elif teacher_type == "immi_diff":
        print("Unsuported teacher for CIFAR-10 (%s) !" % teacher_type)
        sys.exit()
    else:
        print("Unrecognized teacher !")
        sys.exit()

    # Passage des données sous pytorch
    X = th.Tensor(data[:nb_example])
    y = th.Tensor(labels[:nb_example]).view(-1)
    X_test = th.Tensor(data[nb_example:nb_example + nb_test])
    y_test = th.Tensor(labels[nb_example:nb_example + nb_test]).view(-1)
    print(X_test.size())
    sys.stdout.flush()

    batch_size = 32
    nb_batch = int(nb_example / batch_size)

    accuracies = []

    # Boucle d'apprentissage du teacher
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

    # Séléction du student
    if teacher_type == "omni":
        student = OmniscientConvStudent(eta)
    elif teacher_type == "surro_same":
        student = SurrogateConvStudent(eta)
    elif teacher_type == "immi_same":
        student = BaseConv(eta)
    elif teacher_type == "surro_diff":
        print("Unsuported teacher for CIFAR-10 (%s) !" % teacher_type)
        sys.exit()
    elif teacher_type == "immi_diff":
        print("Unsuported teacher for CIFAR-10 (%s) !" % teacher_type)
        sys.exit()
    else:
        print("Unrecognized teacher !")
        sys.exit()

    # Création de l'exemple
    example = BaseConv(eta)

    # Copie des poids des convolution du teacher vers le student et l'exemple
    example.seq = copy.deepcopy(teacher.seq)
    student.seq = copy.deepcopy(teacher.seq)

    example.optim = th.optim.SGD(example.lin.parameters(), lr=example.eta)
    student.optim = th.optim.SGD(student.lin.parameters(), lr=student.eta)

    # 400 itération
    T = 400

    # On séléctionne une sous partie des exemple, trop long sinon
    nb_example = 200
    X = th.Tensor(data[:nb_example])
    y = th.Tensor(labels[:nb_example]).view(-1)
    X_test = th.Tensor(data[nb_example:nb_example + nb_test])
    y_test = th.Tensor(labels[nb_example:nb_example + nb_test]).view(-1)

    batch_size = 1
    nb_batch = int(nb_example / batch_size)

    res_example = []

    # Entrainement de l'exemple
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

    # Entrainement du student avec le teacher
    for t in range(T):
        i = teacher.select_example(student, X.cuda(), y.cuda(), batch_size)

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
    plt.plot(res_student, c='r', label="%s & CNN" % teacher_name)
    plt.title("Cifar10 CNN (class : " + d[class_1] + ", " + d[class_2] + ")")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
