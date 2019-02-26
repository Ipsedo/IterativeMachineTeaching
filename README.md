# IterativeMachineTeaching

Projet de l'option 6 Apprentissage Avancé sur l'article [Iterative Machine Teaching](https://arxiv.org/pdf/1705.10470.pdf).

Malik Kazi-Aoual, David Biard, Samuel Berrien, Nouredine Nour

## Usage

```bash
$ cd /path/to/IterativeMachineTeaching
$ python main.py [experience] [teacher]
```

experience : `gaussian`, `mnist`, `cifar`

teacher : `omni`, `surro_same`, `surro_diff`, `immi`

__Exemple :__


omniscient teacher
```bash
$ python main.py gaussian omni
```
surrogate teacher (same feature space)
```bash
$ python main.py gaussian surro_same
```
surrogate teacher (different feature space)
```bash
$ python main.py gaussian surro_diff
```