import teachers.omniscient_teacher as omni
import numpy as np
import models.neural_network as nn

if __name__ == "__main__":
    X1 = np.random.multivariate_normal([0, 0], np.identity(2), 1000)
    y1 = np.ones((1000,))

    X2 = np.random.multivariate_normal([1.4, 1.4], np.identity(2), 1000)
    y2 = np.zeros((1000,))

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    # On entraine le teacher
    sl = nn.SingleLayer(2, 2)
    epoch = 10
    for e in range(epoch):
        for data, label in zip(X, y):
            gbw = sl.compute_gradient(data.reshape(1, -1), label)
            sl.step(gbw)
        res = np.argmax(sl.forward(X), axis=1)
        nb_correct = (res == y).sum()
        print(nb_correct, "/", X.shape[0])

    # test student
    omni_sl = omni.OmniscientSingleLayer(2, 2)
    print(omni_sl.example_difficulty(X[None, 0], y[None, 0]))
    print(omni_sl.example_use_fullness(sl.weights, X[None, 0], y[None, 0]))

    """
    for e in range(2):
        for data, label in zip(X, y):
            gbw = omni_sl.compute_gradient(data.reshape(1, -1), label)
            omni_sl.step(gbw)
    print(omni_sl.example_difficulty(X[None, 0], y[None, 0]))
    """
