import torch as th
from teachers.utils import *
import sys


def __example_difficulty__(student, X, y):
    student.train()
    student.optim.zero_grad()
    student.lin.weight.retain_grad()
    out = student(X)
    loss = student.loss_fn(out, y)
    loss.backward()
    res = student.lin.weight.grad
    return (th.norm(res) ** 2).item()


def __example_usefulness__(student, loss_teacher, X, y):
    out = student(X)
    loss = student.loss_fn(out, y)
    return loss - loss_teacher


class SurrogateLinearStudent(BaseLinear):
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, loss_teacher, X, y):
        return __example_usefulness__(self, loss_teacher, X, y)


class SurrogateLinearTeacher(BaseLinear):
    def select_example(self, student, X, y, batch_size):

        nb_batch = int(X.size(0) / batch_size)

        min_score = sys.float_info.max
        arg_min = 0
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            data = X[i_min:i_max]
            label = y[i_min:i_max]
            eta = student.optim.param_groups[0]["lr"]
            s = (eta ** 2) * student.example_difficulty(data, label)
            loss_teacher = self.loss_fn(self(data), label)
            s -= eta * 2 * student.example_usefulness(loss_teacher, data, label)
            if s.item() < min_score:
                min_score = s.item()
                arg_min = i
        return arg_min


class SurrogateDiffLinearTeacher(SurrogateLinearTeacher):
    def __init__(self, feature_space, used_feature_space, normal_dist=True):
        super(SurrogateDiffLinearTeacher, self).__init__(used_feature_space)
        if normal_dist:
            self.proj_mat = th.randn(feature_space, used_feature_space)
        else:
            self.proj_mat = th.rand(feature_space, used_feature_space)

    def forward(self, x):
        return super(SurrogateLinearTeacher, self).forward(th.matmul(x, self.proj_mat))
