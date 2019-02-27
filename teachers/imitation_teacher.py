from teachers.utils import BaseLinear, BaseConv
import sys
import torch as th
import copy


def __update_v__(teacher, student, x, is_same_feature_space):
    batch_size = x.size(0)

    for i in range(batch_size):
        data = x[i]
        out_student, out_v = student(data).data, teacher.v(data).data

        eta = student.optim.param_groups[0]["lr"]

        if is_same_feature_space and not hasattr(teacher, "seq"):
            tmp = x[i]
        elif hasattr(teacher, "seq"):
            # Convolution teacher
            tmp = teacher.seq(data.unsqueeze(0)).view(-1)
        else:
            tmp = th.matmul(x[i], teacher.proj_mat)
        teacher.v.lin.weight.data = teacher.v.lin.weight.data - eta * (out_v - out_student) * tmp


def __select_example__(teacher, student, X, y, batch_size, is_same_feature_space):
    __update_v__(teacher, student, teacher.x_t_moins_un, is_same_feature_space)

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

        loss_student = teacher.v.loss_fn(student(data), label)

        teacher.v.train()
        teacher.v.zero_grad()

        # Pour retenir le graphe de calcul de v
        loss_v = teacher.v.loss_fn(teacher.v(data), label)
        loss_v.backward(retain_graph=True)
        teacher.v.zero_grad()
        teacher.v.lin.weight.retain_grad()

        # On applique la loss du student au graphe du gradient de v
        loss_v.data = loss_student.data
        loss_v.backward()

        example_difficulty = (th.norm(teacher.v.lin.weight.grad) ** 2).item()

        teacher.v.train()
        teacher.v.zero_grad()
        teacher.v.lin.weight.retain_grad()

        loss = teacher.v.loss_fn(teacher.v(data), label)

        loss.backward()
        res = teacher.v.lin.weight.grad

        example_usefulness = th.dot(teacher.v.lin.weight.view(-1) - teacher.lin.weight.view(-1), res.view(-1)).item()

        s = eta ** 2 * example_difficulty - 2 * eta * example_usefulness

        if s < min_score:
            min_score = s
            arg_min = i

    teacher.x_t_moins_un = X[arg_min * batch_size:(arg_min + 1) * batch_size]

    return arg_min


class ImitationLinearTeacher(BaseLinear):
    def __init__(self, feature_space, fst_x):
        super(ImitationLinearTeacher, self).__init__(feature_space)
        self.v = BaseLinear(feature_space)
        self.x_t_moins_un = fst_x.view(-1, feature_space)

    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size, True)


class BaseLinearDifferentFeatureSpace(BaseLinear):
    def __init__(self, feature_space, used_feature_space, normal_dist=True, proj_mat=None):
        super(BaseLinearDifferentFeatureSpace, self).__init__(used_feature_space)
        if proj_mat is not None:
            self.proj_mat = proj_mat
        elif normal_dist:
            self.proj_mat = th.randn(feature_space, used_feature_space)
        else:
            self.proj_mat = th.rand(feature_space, used_feature_space)

    def forward(self, x):
        return super(BaseLinearDifferentFeatureSpace, self).forward(th.matmul(x, self.proj_mat))


class ImitationDiffLinearTeacher(BaseLinearDifferentFeatureSpace):
    def __init__(self, feature_space, used_feature_space, fst_x, normal_dist=True):
        super(ImitationDiffLinearTeacher, self).__init__(feature_space, used_feature_space, normal_dist)
        self.v = BaseLinearDifferentFeatureSpace(feature_space, used_feature_space, normal_dist, self.proj_mat)
        self.x_t_moins_un = fst_x.view(-1, feature_space)

    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size, False)


class ImmitationConvTeacher(BaseConv):
    def __init__(self, eta, fst_x):
        super(ImmitationConvTeacher, self).__init__(eta)
        self.v = BaseConv(eta)
        self.x_t_moins_un = fst_x
        self.fst_select_example = True

    def select_example(self, student, X, y, batch_size):
        if self.fst_select_example:
            self.v.seq = copy.deepcopy(self.seq)
            self.fst_select_example = False
        return __select_example__(self, student, X, y, batch_size, True)
