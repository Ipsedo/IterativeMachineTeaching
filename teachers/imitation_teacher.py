from teachers.utils import BaseLinear
import sys
import torch as th


class BaseLinearDifferentFeatureSpace(BaseLinear):
    def __init__(self, feature_space, used_feature_space, normal_dist=True):
        super(BaseLinearDifferentFeatureSpace, self).__init__(used_feature_space)
        if normal_dist:
            self.proj_mat = th.randn(feature_space, used_feature_space)
        else:
            self.proj_mat = th.rand(feature_space, used_feature_space)

    def forward(self, x):
        return super(BaseLinearDifferentFeatureSpace, self).forward(th.matmul(x, self.proj_mat))


class ImitationLinearTeacher(BaseLinearDifferentFeatureSpace):
    def __init__(self, feature_space, used_feature_space, fst_x, normal_dist=True):
        super(ImitationLinearTeacher, self).__init__(feature_space, used_feature_space, normal_dist)
        self.v = BaseLinearDifferentFeatureSpace(feature_space, used_feature_space, normal_dist)
        self.x_t_moins_un = fst_x.view(-1, feature_space)

    def __update_v__(self, student, x):
        batch_size = x.size(0)

        for i in range(batch_size):
            out_student = student(x[i]).data
            out_v = self.v(x[i]).data
            eta = student.optim.param_groups[0]["lr"]
            tmp = th.matmul(x[i], self.proj_mat)
            self.v.lin.weight.data = self.v.lin.weight.data - eta * (out_v - out_student) * tmp

    def select_example(self, student, X, y, batch_size):
        self.__update_v__(student, self.x_t_moins_un)

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

            loss_student = self.v.loss_fn(student(data), label)

            self.v.train()
            self.v.zero_grad()

            # Pour retenir le ggraphe de calcul de v
            loss_v = self.v.loss_fn(self.v(data), label)
            loss_v.backward(th.Tensor(0), retain_graph=True)
            self.v.zero_grad()
            self.v.lin.weight.retain_grad()

            # On applique la loss du student au graphe du gradient de v
            # loss_v.data = th.Tensor(0)
            loss_v.backward(loss_student)

            example_difficulty = (th.norm(self.v.lin.weight.grad) ** 2).item()

            self.v.train()
            self.v.zero_grad()
            self.v.lin.weight.retain_grad()

            loss = self.v.loss_fn(self.v(data), label)

            loss.backward()
            res = self.v.lin.weight.grad

            example_usefulness = th.dot(self.v.lin.weight.view(-1) - self.lin.weight.view(-1), res.view(-1)).item()

            s = eta ** 2 * example_difficulty - 2 * eta * example_usefulness

            if s < min_score:
                min_score = s
                arg_min = i

        self.x_t_moins_un = X[arg_min * batch_size:(arg_min + 1) * batch_size]

        return arg_min
