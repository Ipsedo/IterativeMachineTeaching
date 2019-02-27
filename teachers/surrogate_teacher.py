import torch as th
from teachers.utils import *
import sys


def __example_difficulty__(student, X, y):
    """
    Voir doc teachers.omnsicient_teacher.__example_difficulty__
    :param student:
    :param X:
    :param y:
    :return:
    """
    # Voir explication dans teachers.omnsicient_teacher.__example_difficulty__
    student.train()
    student.optim.zero_grad()
    student.lin.weight.retain_grad()
    out = student(X)
    loss = student.loss_fn(out, y)
    loss.backward()
    res = student.lin.weight.grad
    return (th.norm(res) ** 2).item()


def __example_usefulness__(student, teacher, X, y):
    """
    Calcule l'utilité de l'exemple (X, y) via la différence des loss entre student et teacher
    :param student: modèle du student
    :param loss_teacher: la valeur de la loss du teacher pour l'exemple (X, y)
    :param X: La donnée, ou le batch de données
    :param y: Le(s) label(s) de la donnée ou du batch
    :return: le score d'utilité de l'exemple (X, y)
    """
    # calcul du résultat du student pour X
    out = student(X)

    # calcul de la fonction de perte du student
    loss = student.loss_fn(out, y)

    # calcule de la fonction de perte du teacher
    loss_teacher = teacher.loss_fn(teacher(X), y)

    # retourne la différence entre la loss du student et celle du teacher
    return loss - loss_teacher


def __select_example__(teacher, student, X, y, batch_size):
    """
    Voir doc teachers.omnsicient_teacher.__select_example__
    :param teacher:
    :param student:
    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    # Voir explication dans teachers.omnsicient_teacher.__select_example__
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
        s -= eta * 2 * student.example_usefulness(teacher, data, label)

        if s.item() < min_score:
            min_score = s.item()
            arg_min = i

    return arg_min


class SurrogateLinearStudent(BaseLinear):
    """
    Classe pour le student du surrogate teacher (même ou différent espace de feature)
    """
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, loss_teacher, X, y):
        return __example_usefulness__(self, loss_teacher, X, y)


class SurrogateLinearTeacher(BaseLinear):
    """
    Classe pour le surrogate teacher à même espace de features
    """
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)


class SurrogateDiffLinearTeacher(SurrogateLinearTeacher):
    """
    Classe pour le surrogate teacher avec un espace de features différents
    """
    def __init__(self, feature_space, used_feature_space, normal_dist=True):
        """
        Constructeur surrogate teacher (espace feature différent)
        :param feature_space: L'espace de feature des données
        :param used_feature_space: L'espace de feature du teacher.
            Dans lequel le teacher fera son hypothèse
        :param normal_dist: Choisir ou non une matrice de projection selon une loi normale ou uniforme
        """
        super(SurrogateDiffLinearTeacher, self).__init__(used_feature_space)

        # création de la matrice de projection
        if normal_dist:
            self.proj_mat = th.randn(feature_space, used_feature_space)
        else:
            self.proj_mat = th.rand(feature_space, used_feature_space)

    def forward(self, x):
        # on surcharge la méthode forward pour appliquer le changement d'espace de features
        return super(SurrogateLinearTeacher, self).forward(th.matmul(x, self.proj_mat))


class SurrogateConvStudent(BaseConv):
    """
    Classe pour le student du surrogate teacher pour un modèle à convolution
    """
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, loss_teacher, X, y):
        return __example_usefulness__(self, loss_teacher, X, y)


class SurrogateConvTeacher(BaseConv):
    """
    Classe pour le surrogate teacher pour un modèle à convolution
    """
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)
