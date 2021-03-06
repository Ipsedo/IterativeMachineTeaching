# IterativeMachineTeaching

Projet de l'option 6 Apprentissage Avancé sur l'article [Iterative Machine Teaching](https://arxiv.org/pdf/1705.10470.pdf).

Malik Kazi-Aoual, David Biard, Samuel Berrien, Nouredine Nour


## TODO
* argparse
* optimizations
* code refactoring
* new bench
* [without teacher](https://arxiv.org/abs/2006.15339)

## Usage

```bash
$ cd /path/to/IterativeMachineTeaching
$ python main.py [experience] [teacher]
```

experience : 
- `gaussian` : données gaussienne
- `mnist` : données du MNIST
- `cifar` : données du CIFAR-10

teacher : 
- `omni` : omniscient teacher
- `surro_same` : surrogate teacher dans le même espace de features
- `surro_diff` : surrogate teacher dans un espace de features différent
- `immi_same` : immitation teacher dans le même espace de features
- `immi_diff` : immitation teacher dans un espace de features différent

__CIFAR__ : Executer le script `IterativeMachineTeaching/data/donwload_cifar.sh` dans le dossier 
`IterativeMachineTeaching/data` pour télécharger les données ou simplement placer le dossier `cifar-10-batches-py` 
contenant les données du CIFAR dans de la dossier `IterativeMachineTeaching/data`

__Exemple :__

omniscient teacher sur les données gaussiennes
```bash
$ python main.py gaussian omni
```
surrogate teacher (same feature space) sur les images du CIFAR
```bash
$ python main.py cifar surro_same
```
immitation teacher (different feature space) sur les chiffres du MNIST
```bash
$ python main.py mnist immi_diff
```

## Note
Les teachers utilisant un espace de features différent du student ne sont pas disponibles pour le CIFAR
(plus simplement ils ne sont pas disponibles pour les réseaux de neurones à convolutions).

Pour le CIFAR, l'utilisation de CUDA est activée par défaut. Pour lancer sur le cpu il suffit de supprimer tout les `.cuda()` contenus dans les fichiers `experiences/cifar10.py`,  `teachers/immitation_teacher.py` et `teachers/utils.py` avec la commande `sed` par exemple. 