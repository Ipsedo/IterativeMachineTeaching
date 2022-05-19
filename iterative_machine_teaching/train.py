from enum import Enum

import torch as th
from torchmetrics.functional import f1_score

from tqdm import tqdm

import matplotlib.pyplot as plt

from .networks import Classifier, LinearClassifier, ModelWrapper
from .teachers import Teacher, OmniscientTeacher, SurrogateTeacher, ImitationTeacher
from .student import Student, OmniscientStudent, SurrogateStudent, ImitationStudent

from .data import load_mnist


class Kind(Enum):
    OMNISCIENT = "OMNISCIENT"
    SURROGATE = "SURROGATE"
    IMITATION = "IMITATION"

    @staticmethod
    def __constructors():
        return {
            Kind.OMNISCIENT: (OmniscientTeacher, OmniscientStudent),
            Kind.SURROGATE: (SurrogateTeacher, SurrogateStudent),
            Kind.IMITATION: (ImitationTeacher, ImitationStudent),
        }

    def get_teacher(self, clf: Classifier, learning_rate: float, batch_size: int) -> Teacher:
        return Kind.__constructors()[self][0](clf, learning_rate, batch_size)

    def get_student(self, clf: Classifier, learning_rate: float) -> Student:
        return Kind.__constructors()[self][1](clf, learning_rate)


def train_mnist(
        pickle_path: str,
        kind: Kind
) -> None:

    # load data
    x, y = load_mnist(pickle_path)

    num_classes = th.unique(y).size()[0]

    print("nb example", x.size()[0])

    ratio_train = 4. / 5.
    limit_train = int(x.size()[0] * ratio_train)

    x_train = x[:limit_train, :].cuda()
    y_train = y[:limit_train].cuda()

    x_test = x[limit_train:, :].cuda()
    y_test = y[limit_train:].cuda()

    # create models
    student_model = LinearClassifier(784, 10).cuda()
    teacher_model = LinearClassifier(784, 10).cuda()

    # create student and teacher
    learning_rate = 1e-3
    research_batch_size = 128

    student = kind.get_student(student_model, learning_rate)
    teacher = kind.get_teacher(teacher_model, learning_rate, research_batch_size)

    # Train teacher
    nb_epoch_teacher = 25
    batch_size_teacher = 32
    nb_batch_teacher = x_train.size()[0] // batch_size_teacher

    for e in range(nb_epoch_teacher):
        for b_idx in range(nb_batch_teacher):
            i_min = b_idx * batch_size_teacher
            i_max = (b_idx + 1) * batch_size_teacher

            _ = teacher.train(x_train[i_min:i_max], y_train[i_min:i_max])

        out_test = teacher.predict(x_test)
        f1_score_value = f1_score(out_test, y_test, num_classes=num_classes).item()

        print(f"Epoch {e} : F1-Score = {f1_score_value}")

    # For comparison

    # to avoid a lot of compute...
    nb_example_train_student = 8192
    x_train = x_train[:nb_example_train_student]
    y_train = y_train[:nb_example_train_student]

    rounds = 2048
    batch_size = 16
    nb_batch = x_train.size()[0] // batch_size

    # train example
    example = ModelWrapper(
        LinearClassifier(784, 10).cuda(), learning_rate
    )

    print("\nEntrainement de l'exemple")

    batch_index_example = 0
    loss_values_example = []
    metrics_example = []

    for _ in tqdm(range(rounds)):
        b_idx = batch_index_example % nb_batch
        i_min = b_idx * batch_size
        i_max = (b_idx + 1) * batch_size

        loss = example.train(x_train[i_min:i_max], y_train[i_min:i_max])

        loss_values_example.append(loss)

        batch_index_example += 1

        out_test = example.predict(x_test)
        f1_score_value = f1_score(out_test, y_test, num_classes=num_classes).item()

        metrics_example.append(f1_score_value)

    # train student
    print("\nEntrainement du student")
    loss_values_student = []
    metrics_student = []

    for _ in tqdm(range(rounds)):
        selected_x, selected_y = teacher.select_n_examples(
            student, x_train, y_train, batch_size
        )

        loss = student.train(selected_x, selected_y)

        loss_values_student.append(loss)

        out_test = student.predict(x_test)
        f1_score_value = f1_score(out_test, y_test, num_classes=num_classes).item()

        metrics_student.append(f1_score_value)

    plt.plot(loss_values_example, c='cyan', label="example - loss")
    plt.plot(metrics_example, c='blue', label="example - f1 score")

    plt.plot(loss_values_student, c='magenta', label="student - loss")
    plt.plot(metrics_student, c='red', label="student - f1 score")

    plt.title(f"MNIST Linear ({kind})")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
