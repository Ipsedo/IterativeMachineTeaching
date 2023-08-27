# -*- coding: utf-8 -*-

import pytest
import torch as th

from iterative_machine_teaching.networks import LinearClassifier
from iterative_machine_teaching.students import (
    ImitationStudent,
    OmniscientStudent,
    SurrogateStudent,
)
from iterative_machine_teaching.teachers import (
    ImitationTeacher,
    OmniscientTeacher,
    SurrogateTeacher,
)


@pytest.mark.parametrize("data_size", [4, 8, 16])
@pytest.mark.parametrize("nb_class", [2, 3, 4])
@pytest.mark.parametrize("nb_example", [16, 32, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("top_n", [2, 4, 8])
def test_omniscient_select_n_example(
    data_size: int, nb_class: int, nb_example: int, batch_size: int, top_n: int
) -> None:
    x = th.randn(nb_example, data_size)
    y = th.randint(nb_class, (nb_example,))

    s_model = LinearClassifier(data_size, nb_class)
    student = OmniscientStudent(s_model, 1e-3)

    t_model = LinearClassifier(data_size, nb_class)
    teacher = OmniscientTeacher(t_model, 1e-3, batch_size)

    top_data, top_label = teacher.select_n_examples(student, x, y, top_n)

    assert top_data.size()[0] == top_n
    assert top_label.size()[0] == top_n

    assert top_data.size(1) == data_size
    assert top_data.size()[1] == x.size()[1]
    assert len(top_data.size()) == 2


@pytest.mark.parametrize("data_size", [4, 8, 16])
@pytest.mark.parametrize("nb_class", [2, 3, 4])
@pytest.mark.parametrize("nb_example", [16, 32, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("top_n", [2, 4, 8])
def test_surrogate_select_n_example(
    data_size: int, nb_class: int, nb_example: int, batch_size: int, top_n: int
) -> None:
    x = th.randn(nb_example, data_size)
    y = th.randint(nb_class, (nb_example,))

    s_model = LinearClassifier(data_size, nb_class)
    student = SurrogateStudent(s_model, 1e-3)

    t_model = LinearClassifier(data_size, nb_class)
    teacher = SurrogateTeacher(t_model, 1e-3, batch_size)

    top_data, top_label = teacher.select_n_examples(student, x, y, top_n)

    assert top_data.size()[0] == top_n
    assert top_label.size()[0] == top_n

    assert top_data.size(1) == data_size
    assert top_data.size()[1] == x.size()[1]
    assert len(top_data.size()) == 2


@pytest.mark.parametrize("data_size", [4, 8, 16])
@pytest.mark.parametrize("nb_class", [2, 3, 4])
@pytest.mark.parametrize("nb_example", [16, 32, 64])
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("top_n", [2, 4, 8])
def test_imitation_select_n_example(
    data_size: int, nb_class: int, nb_example: int, batch_size: int, top_n: int
) -> None:
    x = th.randn(nb_example, data_size)
    y = th.randint(nb_class, (nb_example,))

    s_model = LinearClassifier(data_size, nb_class)
    student = ImitationStudent(s_model, 1e-3)

    t_model = LinearClassifier(data_size, nb_class)
    teacher = ImitationTeacher(t_model, 1e-3, batch_size)

    top_data, top_label = teacher.select_n_examples(student, x, y, top_n)

    assert top_data.size()[0] == top_n
    assert top_label.size()[0] == top_n

    assert top_data.size(1) == data_size
    assert top_data.size()[1] == x.size()[1]
    assert len(top_data.size()) == 2
