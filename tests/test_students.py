# -*- coding: utf-8 -*-
import pytest
import torch as th

from iterative_machine_teaching.networks import LinearClassifier
from iterative_machine_teaching.students import OmniscientStudent


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("nb_class", [2, 3, 4])
@pytest.mark.parametrize("data_size", [4, 8, 16])
def test_example_difficulty(
    batch_size: int, nb_class: int, data_size: int
) -> None:
    x = th.randn(batch_size, data_size)
    y = th.randint(nb_class, (batch_size,))

    model = LinearClassifier(data_size, nb_class)
    student = OmniscientStudent(model, 1e-3)

    out = student.example_difficulty(x, y)

    assert out.size()[0] == batch_size
    assert out.size()[0] == x.size()[0]
    assert out.size()[0] == y.size()[0]

    assert len(out.size()) == 1


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("nb_class", [2, 3, 4])
@pytest.mark.parametrize("data_size", [4, 8, 16])
def test_example_usefulness(
    batch_size: int, nb_class: int, data_size: int
) -> None:
    x = th.randn(batch_size, data_size)
    y = th.randint(nb_class, (batch_size,))

    model = LinearClassifier(data_size, nb_class)
    student = OmniscientStudent(model, 1e-3)

    teacher = LinearClassifier(data_size, nb_class)

    out = student.example_usefulness(teacher, x, y)

    assert out.size()[0] == batch_size
    assert out.size()[0] == x.size()[0]
    assert out.size()[0] == y.size()[0]

    assert len(out.size()) == 1
