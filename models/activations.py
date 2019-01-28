import numpy as np


def softmax(z):
    a=z-np.max(z)
    e=np.exp(a)
    s=e.sum(axis=1).reshape(-1,1)
    return e / s


def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0.0, 1.0, 0.0)
    return np.maximum(x, 0.0)


def sigmoid(x, derivative=False):
    if derivative:
        return x*(1-x) # Dérivé
    else:
        return 1./(1.+np.exp(-x))
