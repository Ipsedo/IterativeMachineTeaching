# -*- coding: utf-8 -*-
import copy
from abc import ABC, abstractmethod
from typing import Tuple

import torch as th
import torch.autograd as th_autograd
from torch import nn

from .networks import Classifier, ModelWrapper
from .students import Student


class Teacher(ABC, ModelWrapper):
    def __init__(self, clf: Classifier, learning_rate: float, batch_size: int):
        super().__init__(clf, learning_rate)

        self._batch_size = batch_size

    @abstractmethod
    def select_n_examples(
        self, student: Student, x: th.Tensor, y: th.Tensor, n: int
    ) -> Tuple[th.Tensor, th.Tensor]:
        pass

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clf={self._clf},"
            f"batch_size={self._batch_size})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class OmniscientTeacher(Teacher):
    def select_n_examples(
        self, student: Student, x: th.Tensor, y: th.Tensor, n: int
    ) -> Tuple[th.Tensor, th.Tensor]:
        device = "cuda" if x.is_cuda else "cpu"
        example_scores = th.empty(x.size()[0], device=device)

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
            scores = (eta**2) * student.example_difficulty(
                data, label
            ) - eta * 2 * student.example_usefulness(self._clf, data, label)

            example_scores[i_min:i_max] = scores

        # du plus petit au plus grand
        example_scores = th.argsort(example_scores)

        # on récupère le top N
        top_n = example_scores[:n]

        # et on retourne les exemples associés
        return th.index_select(x, 0, top_n), th.index_select(y, 0, top_n)


class SurrogateTeacher(OmniscientTeacher):
    # Same as omniscient teacher
    pass


class ImitationTeacher(Teacher):
    def __init__(
        self, clf: Classifier, learning_rate: float, research_batch_size: int
    ):
        super().__init__(clf, learning_rate, research_batch_size)

        self.__last_n_examples: th.Tensor = th.empty((0,))

        self.__imitation = copy.deepcopy(clf)

        self.__loss_fn = nn.CrossEntropyLoss(reduction="none")

    def select_n_examples(
        self, student: Student, x: th.Tensor, y: th.Tensor, n: int
    ) -> Tuple[th.Tensor, th.Tensor]:
        if self.__last_n_examples.size(0) == 0:
            self.__last_n_examples = x[:n]

        self.__update_imitation(student)

        nb_batch = x.size()[0] // self._batch_size
        device = "cuda" if x.is_cuda else "cpu"

        example_scores = th.empty(x.size()[0], device=device)

        for b_idx in range(nb_batch):
            # récupération des indices du batch
            i_min = b_idx * self._batch_size
            i_max = (b_idx + 1) * self._batch_size

            # données et labels du batch
            data = x[i_min:i_max]
            label = y[i_min:i_max]

            eta = student.get_eta()

            # Example difficulty
            example_difficulty = self.__example_difficulty(
                student, data, label
            )

            # Example usefulness
            example_usefulness = self.__example_usefulness(data, label)

            score = (
                eta**2 * example_difficulty - 2 * eta * example_usefulness
            )

            example_scores[i_min:i_max] = score

        # du plus petit au plus grand
        example_scores = th.argsort(example_scores)

        # on récupère le top N
        top_n = example_scores[:n]

        # et on retourne les exemples associés
        top_x, top_y = th.index_select(x, 0, top_n), th.index_select(
            y, 0, top_n
        )

        self.__last_n_examples = top_x

        return top_x, top_y

    def __update_imitation(self, student: Student) -> None:
        out_student = student.predict(self.__last_n_examples)
        out_imitation = self.__imitation(self.__last_n_examples)

        eta = self.get_eta()

        self.__imitation.linear.weight.data = (
            self.__imitation.linear.weight.data
            -
            # mean over batch dim
            eta
            * (
                (out_imitation - out_student)[:, :, None]
                * self.__last_n_examples[:, None, :]
            ).mean(dim=0)
        )

    def __get_imitation_gradient(
        self, loss: th.Tensor, device: str
    ) -> th.Tensor:
        return th_autograd.grad(
            outputs=loss,
            inputs=self.__imitation.linear.weight,
            grad_outputs=(th.eye(self._batch_size, device=device),),
            create_graph=True,
            retain_graph=True,
            is_grads_batched=True,
        )[0]

    def __example_difficulty(
        self, student: Student, data: th.Tensor, label: th.Tensor
    ) -> th.Tensor:
        device = "cuda" if next(self._clf.parameters()).is_cuda else "cpu"

        loss_student = self.__loss_fn(student.predict(data), label)
        loss_imitation = self.__loss_fn(self.__imitation(data), label)

        loss_imitation.data = loss_student.data

        # get teacher gradient
        imitation_grad = self.__get_imitation_gradient(loss_imitation, device)

        difficulty: th.Tensor = (
            th.norm(imitation_grad.view(self._batch_size, -1), dim=1) ** 2
        )

        return difficulty

    def __example_usefulness(
        self, data: th.Tensor, label: th.Tensor
    ) -> th.Tensor:
        device = "cuda" if next(self._clf.parameters()).is_cuda else "cpu"

        loss_imitation = self.__loss_fn(self.__imitation(data), label)

        imitation_grad = self.__get_imitation_gradient(loss_imitation, device)

        usefulness: th.Tensor = (
            (
                self.__imitation.linear.weight.view(1, -1)
                - self._clf.linear.weight.view(1, -1)
            )
            * imitation_grad.view(self._batch_size, -1)
        ).sum(dim=1)

        return usefulness
