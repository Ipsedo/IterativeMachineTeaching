# -*- coding: utf-8 -*-
from enum import Enum
from typing import Dict, NamedTuple, Tuple, Type

import matplotlib.pyplot as plt
import torch as th
from torchmetrics.functional import f1_score
from tqdm import tqdm

from .networks import Classifier, LinearClassifier, ModelWrapper
from .students import (
    ImitationStudent,
    OmniscientStudent,
    Student,
    SurrogateStudent,
)
from .teachers import (
    ImitationTeacher,
    OmniscientTeacher,
    SurrogateTeacher,
    Teacher,
)

DatasetOptions = NamedTuple(
    "DatasetOptions",
    [
        ("name", str),
        ("x", th.Tensor),
        ("y", th.Tensor),
        ("train_ratio", float),
    ],
)

StudentOptions = NamedTuple(
    "StudentOptions",
    [
        ("examples", int),
        ("steps", int),
        ("batch_size", int),
        ("learning_rate", float),
    ],
)

TeacherOptions = NamedTuple(
    "TeacherOptions",
    [
        ("learning_rate", float),
        ("batch_size", int),
        ("research_batch_size", int),
        ("nb_epoch", int),
    ],
)


class TeachingType(Enum):
    OMNISCIENT = "OMNISCIENT"
    SURROGATE = "SURROGATE"
    IMITATION = "IMITATION"

    @staticmethod
    def __constructors() -> Dict[
        "TeachingType", Tuple[Type[Teacher], Type[Student]]
    ]:
        return {
            TeachingType.OMNISCIENT: (OmniscientTeacher, OmniscientStudent),
            TeachingType.SURROGATE: (SurrogateTeacher, SurrogateStudent),
            TeachingType.IMITATION: (ImitationTeacher, ImitationStudent),
        }

    def get_teacher(
        self, clf: Classifier, learning_rate: float, batch_size: int
    ) -> Teacher:
        return TeachingType.__constructors()[self][0](
            clf, learning_rate, batch_size
        )

    def get_student(self, clf: Classifier, learning_rate: float) -> Student:
        return TeachingType.__constructors()[self][1](clf, learning_rate)


def train(
    dataset_options: DatasetOptions,
    kind: TeachingType,
    teacher_options: TeacherOptions,
    student_options: StudentOptions,
    cuda: bool,
) -> None:

    assert 0.0 < dataset_options.train_ratio < 1.0

    x, y = dataset_options.x, dataset_options.y

    num_features = x.size()[1]
    num_classes = th.unique(y).size()[0]

    print(
        f'Dataset "{dataset_options.name}" of {x.size()[0]} '
        f"examples with {kind.value} teacher."
    )

    limit_train = int(x.size()[0] * dataset_options.train_ratio)

    x_train = x[:limit_train, :]
    y_train = y[:limit_train]

    x_test = x[limit_train:, :]
    y_test = y[limit_train:]

    # create models
    student_model = LinearClassifier(num_features, num_classes)
    example_model = LinearClassifier(num_features, num_classes)
    teacher_model = LinearClassifier(num_features, num_classes)

    # cuda or not
    if cuda:
        x_train = x_train.cuda()
        y_train = y_train.cuda()

        x_test = x_test.cuda()
        y_test = y_test.cuda()

        student_model = student_model.cuda()
        example_model = example_model.cuda()
        teacher_model = teacher_model.cuda()

    # create student, example and teacher
    student = kind.get_student(student_model, student_options.learning_rate)
    example = ModelWrapper(example_model, student_options.learning_rate)
    teacher = kind.get_teacher(
        teacher_model,
        teacher_options.learning_rate,
        teacher_options.research_batch_size,
    )

    # Train teacher
    print("Train teacher...")
    nb_batch_teacher = x_train.size()[0] // teacher_options.batch_size

    tqdm_bar = tqdm(range(teacher_options.nb_epoch))
    for e in tqdm_bar:
        for b_idx in range(nb_batch_teacher):
            i_min = b_idx * teacher_options.batch_size
            i_max = (b_idx + 1) * teacher_options.batch_size

            _ = teacher.train(x_train[i_min:i_max], y_train[i_min:i_max])

        out_test = teacher.predict(x_test)
        f1_score_value = f1_score(
            out_test, y_test, num_classes=num_classes, task="multiclass"
        ).item()

        tqdm_bar.set_description(f"Epoch {e} : F1-Score = {f1_score_value}")

    # For benchmark

    # to avoid a lot of compute...
    # if negative -> all train examples
    student_examples = (
        student_options.examples
        if student_options.examples >= 0
        else x_train.size()[0]
    )
    x_train = x_train[:student_examples]
    y_train = y_train[:student_examples]

    nb_batch = x_train.size()[0] // student_options.batch_size

    # train example
    print("Train example...")

    batch_index_example = 0
    loss_values_example = []
    metrics_example = []

    for _ in tqdm(range(student_options.steps)):
        b_idx = batch_index_example % nb_batch
        i_min = b_idx * student_options.batch_size
        i_max = (b_idx + 1) * student_options.batch_size

        loss = example.train(x_train[i_min:i_max], y_train[i_min:i_max])

        loss_values_example.append(loss)

        batch_index_example += 1

        out_test = example.predict(x_test)
        f1_score_value = f1_score(
            out_test, y_test, num_classes=num_classes, task="multiclass"
        ).item()

        metrics_example.append(f1_score_value)

    # train student
    print("Train student...")

    loss_values_student = []
    metrics_student = []

    for _ in tqdm(range(student_options.steps)):
        selected_x, selected_y = teacher.select_n_examples(
            student, x_train, y_train, student_options.batch_size
        )

        loss = student.train(selected_x, selected_y)

        loss_values_student.append(loss)

        out_test = student.predict(x_test)
        f1_score_value = f1_score(
            out_test, y_test, num_classes=num_classes, task="multiclass"
        ).item()

        metrics_student.append(f1_score_value)

    plt.plot(loss_values_example, c="cyan", label="example - loss")
    plt.plot(loss_values_student, c="magenta", label="student - loss")

    plt.plot(metrics_example, c="blue", label="example - f1 score")
    plt.plot(metrics_student, c="red", label="student - f1 score")

    plt.title(f"{dataset_options.name} Linear - {kind.value}")
    plt.xlabel("mini-batch optim steps")
    plt.legend()
    plt.show()
