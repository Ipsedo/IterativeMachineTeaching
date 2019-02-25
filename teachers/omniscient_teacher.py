from teachers.utils import *
import torch as th
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


def __example_usefulness__(student, w_star, X, y):
    student.train()
    student.optim.zero_grad()
    student.lin.weight.retain_grad()
    out = student(X)
    loss = student.loss_fn(out, y)
    loss.backward()
    res = student.lin.weight.grad
    diff = student.lin.weight - w_star
    return th.dot(diff.view(-1), res.view(-1)).item()


def __select_example__(teacher, student, X, y, batch_size):
    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    min_score = sys.float_info.max
    arg_min = 0
    for i in range(nb_batch):
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        data = X[i_min:i_max]
        label = y[i_min:i_max]
        eta = student.optim.param_groups[0]["lr"]
        s = (eta ** 2) * student.example_difficulty(data, label)
        s -= eta * 2 * student.example_usefulness(teacher.lin.weight, data, label)
        if s < min_score:
            min_score = s
            arg_min = i

    return arg_min


class OmniscientLinearStudent(BaseLinear):
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, w_star, X, y):
        return __example_usefulness__(self, w_star, X, y)


class OmniscientConvStudent(BaseConv):
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, w_star, X, y):
        return __example_usefulness__(self, w_star, X, y)


class OmniscientLinearTeacher(BaseLinear):
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)


class OmniscientConvTeacher(BaseLinear):
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)
