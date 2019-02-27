from teachers.utils import BaseLinear, BaseConv
import torch as th
import sys


def __example_difficulty__(student, X, y):
    """
    Retourne la difficulté de l'exemple (X, y) selon le student
    :param student: Student ayant un attribut "lin" de class torch.nn.Linear
    :param X: La donnée
    :param y: Le label de la donnée
    :return: Le score de difficulté de l'exemple (X, y)
    """
    # On veut pouvoir calculer le gradient -> train()
    student.train()

    # Mise à zéro du gradient accumulé sur les poids du student
    student.optim.zero_grad()

    # On veut retenir le gradient des poids de la couche linéaire lin
    student.lin.weight.retain_grad()

    # récupération de la sortie du student
    out = student(X)

    # calcul de la fonction de perte
    loss = student.loss_fn(out, y)

    # calcul du gradient
    loss.backward()

    # récupération du gradient de la couche
    res = student.lin.weight.grad

    # retourne la norme du gradient au carré
    return (th.norm(res) ** 2).item()


def __example_usefulness__(student, w_star, X, y):
    """
    Retourne l'utilité de l'exemple (X, y) selon le student et les poids du teacher
    :param student: Student ayant un attribut "lin" de class torch.nn.Linear
    :param w_star: Les poids du teacher (hypothèse  objectif)
    :param X: La donnée
    :param y: Le label de la donnée
    :return: Le score d'utilité de l'exemple (X, y)
    """
    # On veut pouvoir calculer le gradient -> train()
    student.train()

    # Mise à zéro du gradient accumulé sur les poids du student
    student.optim.zero_grad()

    # On veut retenir le gradient des poids de la couche linéaire lin
    student.lin.weight.retain_grad()

    # récupération de la sortie du student
    out = student(X)

    # calcul de la fonction de perte
    loss = student.loss_fn(out, y)

    # calcul du gradient
    loss.backward()

    # récupération du gradient de la couche
    res = student.lin.weight.grad

    # différence des poids entre le student et le teacher
    diff = student.lin.weight - w_star

    # produit scalaire entre la différence des poids et le gradient du student
    return th.dot(diff.view(-1), res.view(-1)).item()


def __select_example__(teacher, student, X, y, batch_size):
    """
    Selectionne un exemple selon le teacher et le student
    :param teacher: Le teacher de classe mère BaseLinear
    :param student: Le student devant implémenter les deux méthodes example_difficulty et example_usefulness
    :param X: Les données
    :param y: les labels des données
    :param batch_size: La taille d'un batch de données
    :return: L'indice de l'exemple à enseigner au student
    """
    nb_example = X.size(0)
    nb_batch = int(nb_example / batch_size)

    min_score = sys.float_info.max
    arg_min = 0

    for i in range(nb_batch):
        # récupération des indices du batch
        i_min = i * batch_size
        i_max = (i + 1) * batch_size

        # données et labels du batch
        data = X[i_min:i_max]
        label = y[i_min:i_max]

        # taux d'apprentissage
        eta = student.optim.param_groups[0]["lr"]

        # Caclul du score du batch
        s = (eta ** 2) * student.example_difficulty(data, label)
        s -= eta * 2 * student.example_usefulness(teacher.lin.weight, data, label)

        # MAJ du meilleur score (ie le plus petit score)
        if s < min_score:
            min_score = s
            arg_min = i

    return arg_min


class OmniscientLinearStudent(BaseLinear):
    """
    Classe pour le student du omniscient teacher
    Classification linéaire
    Marche de paire avec OmniscientLinearTeacher
    """
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, w_star, X, y):
        return __example_usefulness__(self, w_star, X, y)


class OmniscientConvStudent(BaseConv):
    """
    Classe pour le student du omniscient teacher
    Modèle à convolution.
    Marche de paire avec OmniscientConvTeacher
    """
    def example_difficulty(self, X, y):
        return __example_difficulty__(self, X, y)

    def example_usefulness(self, w_star, X, y):
        return __example_usefulness__(self, w_star, X, y)


class OmniscientLinearTeacher(BaseLinear):
    """
    Omniscient teacher.
    Pour un classifieur linéaire de classe OmniscientLinearStudent
    """
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)


class OmniscientConvTeacher(BaseConv):
    """
    Omnsicient teacher
    Pour un classifieur à convolution de classe OmniscientConvStudent
    """
    def select_example(self, student, X, y, batch_size):
        return __select_example__(self, student, X, y, batch_size)
