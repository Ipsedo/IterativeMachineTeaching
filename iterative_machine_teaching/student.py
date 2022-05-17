from abc import ABC, abstractmethod

import torch as th
import torch.nn as nn
import torch.autograd as th_autograd

from .networks import Classifier, ModelWrapper


class Student(ABC, ModelWrapper):
    def __init__(self, clf: Classifier, learning_rate: float):
        super(Student, self).__init__(clf, learning_rate)

        self._loss_fn_reduction_none = nn.MSELoss(reduction='none')

    @abstractmethod
    def example_difficulty(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def example_usefulness(self, teacher: Classifier, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        pass

    def get_eta(self) -> float:
        return self._optim.param_groups[0]["lr"]


class OmniscientStudent(Student):

    def _get_weight_gradient(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        out = self._clf(x)
        out = self._loss_fn_reduction_none(out, y)

        batch_size = x.size()[0]
        device = "cuda" if x.is_cuda else "cpu"

        return th_autograd.grad(
            outputs=out,
            inputs=self._clf.clf.weight,
            grad_outputs=(th.eye(batch_size, device=device),),
            create_graph=True, retain_graph=True,
            is_grads_batched=True
        )[0]

    def example_difficulty(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        gradients_weight = self._get_weight_gradient(x, y)

        batch_size = x.size()[0]

        return th.norm(gradients_weight.view(batch_size, -1), dim=1) ** 2

    def example_usefulness(self, teacher: Classifier, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        gradients_weight = self._get_weight_gradient(x, y)

        batch_size = x.size()[0]

        diff = self._clf.clf.weight - teacher.clf.weight

        return (
                diff[None, :, :].view(1, -1) *
                gradients_weight.view(batch_size, -1)
        ).sum(dim=1)


class SurrogateStudent(OmniscientStudent):
    def example_usefulness(self, teacher: Classifier, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        out_student = self._clf(x)
        out_teacher = teacher(x)

        loss_student = self._loss_fn_reduction_none(out_student, y)
        loss_teacher = self._loss_fn_reduction_none(out_teacher, y)

        return loss_student - loss_teacher

