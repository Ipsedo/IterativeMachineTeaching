# -*- coding: utf-8 -*-
import copy
import sys

import torch as th

from .utils import BaseConv, BaseLinear


def __update_v__(teacher, student, x, is_same_feature_space):
    """
    Met à jour le modèle qui immite le student dans l'espace de feature du teacher
    :param teacher: Le immitation teacher
    :param student: Le student de classe BaseLinear ou BaseConv
    :param x: Le batch de données
    :param is_same_feature_space: Si le teacher partage le même espace de feature que le student
    :return: Rien (procedure)
    """
    batch_size = x.size(0)

    # on itère sur les données du batch
    for i in range(batch_size):
        data = x[i]

        # on récupère la sortie de l'immitation et du student
        out_student, out_v = student(data).data, teacher.v(data).data

        # Le learning rate du student
        eta = student.optim.param_groups[0]["lr"]

        # désambiguisation entre immitation teacher avec même espace de feature ou non,
        # et le immitation teacher avec convolution
        if is_same_feature_space and not hasattr(teacher, "seq"):
            tmp = x[i]
        elif hasattr(teacher, "seq"):
            # Convolution teacher
            tmp = teacher.seq(data.unsqueeze(0)).view(-1)
        else:
            tmp = th.matmul(x[i], teacher.proj_mat)

        # MAJ de l'immitation v en fonction du résultat du student
        teacher.v.lin.weight.data = (
            teacher.v.lin.weight.data - eta * (out_v - out_student) * tmp
        )


def __select_example__(
    teacher, student, X, y, batch_size, is_same_feature_space
):
    """
    Selectionne un exemple dans les données (X, y)
    :param teacher: Le immitation teacher devant avoir un attribut v (de classe mère du teacher)
                    et un attribut x_t_moins_un de type th.Tensor
    :param student: Le student ayant un attribut lin de type torch.nn.Linear
    :param X: Les données
    :param y: Les labels de données
    :param batch_size: La taille de batch
    :param is_same_feature_space: Si le teacher a le même espace de feature que le student
    :return: L'indice de l'exemple à enseigner au student
    """

    # On met à jour l'immitation faite par le teacher avec le dernier exemple choisi
    __update_v__(teacher, student, teacher.x_t_moins_un, is_same_feature_space)

    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    min_score = sys.float_info.max
    arg_min = 0

    # TODO
    # - one "forward" scoring pass
    # - sort n * log(n)
    # - get first examples

    # On itère sur les données
    for i in range(nb_batch):
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        data = X[i_min:i_max]
        label = y[i_min:i_max]

        eta = student.optim.param_groups[0]["lr"]

        # calcul de la loss selon la sortie du student
        loss_student = teacher.v.loss_fn(student(data), label)

        teacher.v.train()
        teacher.v.zero_grad()

        # Pour retenir le graphe de calcul de v
        # PyTorch a besoin d'un graphe de calcul pour appliquer calculer le gradient
        loss_v = teacher.v.loss_fn(teacher.v(data), label)
        loss_v.backward(retain_graph=True)
        teacher.v.zero_grad()
        teacher.v.lin.weight.retain_grad()

        # On backward avec la loss du student sur l'immitation faite par le teacher
        loss_v.data = loss_student.data
        loss_v.backward()

        # On calcule la difficulté de l'exemple avec la norme du gradient calculé avant au carré
        example_difficulty = (th.norm(teacher.v.lin.weight.grad) ** 2).item()

        teacher.v.train()
        teacher.v.zero_grad()
        teacher.v.lin.weight.retain_grad()

        # On calcule la loss de l'immitation faite par le teacher
        loss = teacher.v.loss_fn(teacher.v(data), label)

        # On récupère son gradient
        loss.backward()
        res = teacher.v.lin.weight.grad

        # On calcule le score d'utilité de l'exemple
        example_usefulness = th.dot(
            teacher.v.lin.weight.view(-1) - teacher.lin.weight.view(-1),
            res.view(-1),
        ).item()

        # On calcule le score générale de l'exemple
        s = eta**2 * example_difficulty - 2 * eta * example_usefulness

        # MAJ du meilleur exemple
        if s < min_score:
            min_score = s
            arg_min = i

    # MAJ du précédent exemple choisi
    teacher.x_t_moins_un = X[arg_min * batch_size : (arg_min + 1) * batch_size]

    return arg_min


class ImitationLinearTeacher(BaseLinear):
    """
    Immitation teacher pour un modèle linéaire
    """

    def __init__(self, feature_space, fst_x):
        super(ImitationLinearTeacher, self).__init__(feature_space)
        # définition de v, le modèle que va entrainer le teacher pour immiter le student
        self.v = BaseLinear(feature_space)
        # dernier exemple choisi (premier choisi aléatoirement)
        self.x_t_moins_un = fst_x.view(-1, feature_space)

    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size, True)


class BaseLinearDifferentFeatureSpace(BaseLinear):
    """
    Classe pour un classifieur linéaire n'adoptant pas le même espace de feature que les données
    """

    def __init__(
        self,
        feature_space,
        used_feature_space,
        normal_dist=True,
        proj_mat=None,
    ):
        super(BaseLinearDifferentFeatureSpace, self).__init__(
            used_feature_space
        )

        # définition matrice de projection
        if proj_mat is not None:
            self.proj_mat = proj_mat
        elif normal_dist:
            self.proj_mat = th.randn(feature_space, used_feature_space)
        else:
            self.proj_mat = th.rand(feature_space, used_feature_space)

    def forward(self, x):
        # surcharge méthode -> projection des données
        return super(BaseLinearDifferentFeatureSpace, self).forward(
            th.matmul(x, self.proj_mat)
        )


class ImitationDiffLinearTeacher(BaseLinearDifferentFeatureSpace):
    """
    Immitation teacher avec espace de features différent pour un modèle linéaire.
    Hérite donc de la classe BaseLinearDifferentFeatureSpace
    """

    def __init__(
        self, feature_space, used_feature_space, fst_x, normal_dist=True
    ):
        super(ImitationDiffLinearTeacher, self).__init__(
            feature_space, used_feature_space, normal_dist
        )

        # définition de l'immitation du student
        self.v = BaseLinearDifferentFeatureSpace(
            feature_space, used_feature_space, normal_dist, self.proj_mat
        )
        # et du premier exemple
        self.x_t_moins_un = fst_x.view(-1, feature_space)

    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size, False)


class ImmitationConvTeacher(BaseConv):
    """
    Immitation teacher pour modèle à convolution
    """

    def __init__(self, eta, fst_x):
        super(ImmitationConvTeacher, self).__init__(eta)

        # définition de l'immitation
        self.v = BaseConv(eta)
        # et du premier example
        self.x_t_moins_un = fst_x

        # Pour recopier les poids des convolution du teacher à l'immitation v lors de la premiere requete select_example
        self.fst_select_example = True

    def select_example(self, student, X, y, batch_size):
        # On copie les poids des convolution (contenues dans seq de type nn.Sequential)
        # dans l'immitation v
        if self.fst_select_example:
            self.v.seq = copy.deepcopy(self.seq)
            self.fst_select_example = False

        return __select_example__(self, student, X, y, batch_size, True)
