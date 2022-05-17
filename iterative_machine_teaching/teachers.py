from abc import ABC, abstractmethod
from typing import Tuple

import torch as th

from .networks import Classifier, ModelWrapper
from .student import Student


class Teacher(ABC, ModelWrapper):
    def __init__(self, clf: Classifier, learning_rate: float, batch_size: int):
        super(Teacher, self).__init__(clf, learning_rate)

        self._batch_size = batch_size

    @abstractmethod
    def select_n_examples(self, student: Student, x: th.Tensor, y: th.Tensor, n: int) -> Tuple[th.Tensor, th.Tensor]:
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(" \
               f"clf={self._clf}," \
               f"batch_size={self._batch_size})"

    def __repr__(self):
        return self.__str__()


class OmniscientTeacher(Teacher):

    def select_n_examples(self, student: Student, x: th.Tensor, y: th.Tensor, n: int) -> Tuple[th.Tensor, th.Tensor]:
        example_scores = []

        nb_batch = x.size()[0] // self._batch_size

        for b_idx in range(nb_batch):
            # récupération des indices du batch
            i_min = b_idx * self._batch_size
            i_max = (b_idx + 1) * self._batch_size

            # données et labels du batch
            data = x[i_min:i_max]
            label = y[i_min:i_max]

            eta = student.get_eta()

            # calcul du score
            s = (eta ** 2) * student.example_difficulty(data, label)
            s -= eta * 2 * student.example_usefulness(self._clf, data, label)

            example_scores.append(s)

        example_scores = th.cat(example_scores, dim=0)

        # du plus petit au plus grand
        example_scores = th.argsort(example_scores)

        # on récupère le top N
        top_n = example_scores[:n]

        # et on retourne les exemples associés
        return th.index_select(x, 0, top_n), th.index_select(y, 0, top_n)


class SurrogateTeacher(OmniscientTeacher):
    # Same as omniscient teacher
    pass
