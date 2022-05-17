from enum import Enum

import torch as th

from tqdm import tqdm

import matplotlib.pyplot as plt

from .networks import Clf, LinearClf, ModelWrapper
from .teachers import Teacher, OmniscientTeacher
from .student import Student, OmniscientStudent

from .data import load_mnist


class Kind(Enum):
    OMNISCIENT = "OMNISCIENT"
    SURROGATE = "SURROGATE"
    IMITATION = "IMITATION"

    def get_teacher(self, clf: Clf, learning_rate: float, batch_size: int) -> Teacher:
        # TODO 3.10 switch
        if self == self.OMNISCIENT:
            return OmniscientTeacher(
                clf, learning_rate, batch_size
            )
        else:
            raise NotImplemented(
                "Only omniscient is implemented."
            )

    def get_student(self, clf: Clf, learning_rate: float) -> Student:
        # TODO 3?10 switch
        if self == self.OMNISCIENT:
            return OmniscientStudent(
                clf, learning_rate
            )
        else:
            raise NotImplemented(
                "Only omniscient is implemented."
            )


def train_mnist(
        pickle_path: str,
        first_class: int,
        second_class: int,
        kind: Kind

) -> None:
    assert 0 <= first_class <= 9
    assert 0 <= second_class <= 9
    assert first_class != second_class

    # load data
    x, y = load_mnist(pickle_path)

    mask_class = (y == first_class) | (y == second_class)

    x = x[mask_class, :]
    y = y[mask_class]

    print("nb example", x.size()[0])

    y = th.where(y == first_class, 1, 0).to(th.float)

    test_ratio = 1. / 8.
    limit_train = int((1. - test_ratio) * x.size()[0])

    x_train = x[:limit_train, :]
    y_train = y[:limit_train]

    x_test = x[limit_train:, :]
    y_test = y[limit_train:]

    # create models
    student_model = LinearClf(784)
    teacher_model = LinearClf(784)

    # create student and teacher
    learning_rate = 1e-4

    student = kind.get_student(student_model, learning_rate)
    teacher = kind.get_teacher(teacher_model, learning_rate, 256)

    rounds = 1024

    # Train teacher
    nb_epoch_teacher = 50
    batch_size_teacher = 32
    nb_batch_teacher = x_train.size()[0] // batch_size_teacher

    for e in range(nb_epoch_teacher):
        for b_idx in range(nb_batch_teacher):
            i_min = b_idx * batch_size_teacher
            i_max = (b_idx + 1) * batch_size_teacher

            _ = teacher.train(x_train[i_min:i_max], y_train[i_min:i_max])

        out_test = teacher.predict(x_test)
        out_test = th.where(out_test > 0.5, 1, 0)
        nb_correct = (out_test == y_test).sum().item()
        print(f"Epoch {e} : nb_correct (TP and TN) = {nb_correct} / {x_test.size()[0]}")

    # train example
    batch_size = 8
    nb_batch = x_train.size()[0] // batch_size

    example = ModelWrapper(
        LinearClf(784), learning_rate
    )

    print("\nEntrainement de l'exemple")

    batch_index_example = 0
    loss_values_example = []
    accuracy_example = []

    for _ in tqdm(range(rounds)):
        b_idx = batch_index_example % nb_batch
        i_min = b_idx * batch_size
        i_max = (b_idx + 1) * batch_size

        loss = example.train(x_train[i_min:i_max], y_train[i_min:i_max])

        loss_values_example.append(loss)

        batch_index_example += 1

        out_test = example.predict(x_test)
        out_test = th.where(out_test > 0.5, 1, 0)
        nb_correct = (out_test == y_test).sum().item()
        accuracy_example.append(nb_correct / x_test.size()[0])

    # train student
    print("\nEntrainement du student")
    loss_values_student = []
    accuracy_student = []

    for _ in tqdm(range(rounds)):
        selected_x, selected_y = teacher.select_n_examples(
            student, x_train, y_train, batch_size
        )

        loss = student.train(selected_x, selected_y)

        loss_values_student.append(loss)

        out_test = student.predict(x_test)
        out_test = th.where(out_test > 0.5, 1, 0)
        nb_correct = (out_test == y_test).sum().item()
        accuracy_student.append(nb_correct / x_test.size()[0])

    plt.plot(loss_values_example, c='cyan', label="example - loss")
    plt.plot(accuracy_example, c='blue', label="example - accuracy")

    plt.plot(loss_values_student, c='magenta', label="student - loss")
    plt.plot(accuracy_student, c='red', label="student - accuracy")

    plt.title("MNIST Linear model (class : " + str(first_class) + ", " + str(second_class) + ")")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
