# -*- coding: utf-8 -*-
import gzip
import pickle
import sys

import numpy as np

"""
MiniNN - Minimal Neural Network
This code is a straigthforward and minimal implementation
of a multi-layer neural network for training on MNIST dataset.
It is mainly intended for educational and prototyping purpuses.
"""
__author__ = "Gaetan Marceau Caron (gaetan.marceau-caron@inria.fr)"
__copyright__ = "Copyright (C) 2015 Gaetan Marceau Caron"
__license__ = "CeCILL 2.1"
__version__ = "1.0"


def unpickle(file):
    fo = open(file, "rb")
    dict = pickle.load(fo)
    fo.close()
    return dict


def load_ramp(valid_size=0.8):
    target_column_name = "TARGET"
    df = pd.read_csv("./ramp_data.csv")
    y = df[target_column_name].values
    X = df.drop(target_column_name, axis=1).values

    n_train = int(valid_size * X.shape[0])
    train_perm = np.random.permutation(X.shape[0])
    X = X[train_perm, :]
    y = y[train_perm]
    train_set = [X[:n_train], y[:n_train]]
    valid_set = [X[n_train:], y[n_train:]]

    return train_set, valid_set


def load_mnist(fname="mnist.pkl.gz"):
    f = gzip.open(fname, "rb")

    if sys.version_info.major == 2:
        train_set, valid_set, test_set = pickle.load(
            f
        )  # compatibility issue between python 2.7 and 3.4
    else:
        train_set, valid_set, test_set = pickle.load(
            f, encoding="latin-1"
        )  # compatibility issue between python 2.7 and 3.4
    f.close()

    # Shuffle
    train_X = train_set[0]
    train_y = train_set[1]
    valid_X = valid_set[0]
    valid_y = valid_set[1]

    train_perm = np.random.permutation(train_X.shape[0])
    train_set = [train_X[train_perm, :], train_y[train_perm]]

    valid_perm = np.random.permutation(valid_X.shape[0])
    valid_set = [valid_X[valid_perm, :], valid_y[valid_perm]]

    return train_set, valid_set, test_set


def load_cifar10():
    batch1 = unpickle("./data/cifar-10-batches-py/data_batch_1")
    batch2 = unpickle("./cifar-10-batches-py/data_batch_2")
    batch3 = unpickle("./cifar-10-batches-py/data_batch_3")
    batch4 = unpickle("./cifar-10-batches-py/data_batch_4")
    batch5 = unpickle("./cifar-10-batches-py/data_batch_5")

    test_batch = unpickle("./cifar-10-batches-py/test_batch")

    data = np.vstack(
        (
            batch1["data"],
            batch2["data"],
            batch3["data"],
            batch4["data"],
            batch5["data"],
        )
    )
    labels = np.concatenate(
        (
            batch1["labels"],
            batch2["labels"],
            batch3["labels"],
            batch4["labels"],
            batch5["labels"],
        )
    )

    train_set = [data / 255.0, labels]
    valid_set = [test_batch["data"] / 255.0, np.array(test_batch["labels"])]

    return train_set, valid_set, valid_set


def load_cifar10_2(nbbatch=5):
    all_data = []
    all_labels = []
    for i in range(nbbatch):
        data = open("./data/cifar-10-batches-py/data_batch_%s" % (i + 1), "rb")
        dict = pickle.load(data, encoding="bytes")
        data = dict[b"data"]
        labels = np.asarray(dict[b"labels"]).reshape((-1, 1))
        all_data.append(data)
        all_labels.append(labels)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return (all_data, all_labels)


def cifar10_proper_array(data):
    all_red = data[:, :1024].reshape(-1, 32, 32)
    all_green = data[:, 1024:2048].reshape(-1, 32, 32)
    all_blue = data[:, 2048:].reshape(-1, 32, 32)
    return np.stack([all_red, all_green, all_blue], axis=1) / 255.0


def cifar10_dictclass():
    return {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }


def load_cifar100():
    train_dataset = unpickle("./cifar_100/train")
    test_dataset = unpickle("./cifar_100/test")

    train_data = train_dataset["data"] / 255.0
    train_labels = np.array(train_dataset["fine_labels"])

    test_data = test_dataset["data"] / 255.0
    test_labels = np.array(test_dataset["fine_labels"])

    train_set = [train_data, train_labels]
    valid_set = [test_data, test_labels]

    return train_set, valid_set, valid_set


def load_otto():

    # import data
    train = pd.read_csv("otto_train.csv")
    test = pd.read_csv("otto_test.csv")

    # Shuffling the data
    train = train.loc[np.random.permutation(train.index)]

    # drop ids and get labels
    labels = train.target.values
    train = train.drop("id", axis=1)
    train = train.drop("target", axis=1)
    test = test.drop("id", axis=1)

    # encode labels
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)

    train = np.array(train)
    train = sklearn.preprocessing.normalize(train)

    train_set = []
    train_set.append(train[:50000])
    train_set.append(labels[:50000])

    valid_set = []
    valid_set.append(train[50000:])
    valid_set.append(labels[50000:])

    return train_set, valid_set, valid_set


def load_feat_otto():

    # import data
    train = pd.read_csv("otto_train.csv")
    test = pd.read_csv("otto_test.csv")

    # Shuffling the data
    train = train.loc[np.random.permutation(train.index)]

    # drop ids and get labels
    labels = train.target.values
    train = train.drop("id", axis=1)
    train = train.drop("target", axis=1)
    test = test.drop("id", axis=1)

    # transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer()
    train = tfidf.fit_transform(train).toarray()
    test = tfidf.transform(test).toarray()

    # encode labels
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)

    # train = np.array(train)

    train_set = []
    train_set.append(train[:50000])
    train_set.append(labels[:50000])

    valid_set = []
    valid_set.append(train[50000:])
    valid_set.append(labels[50000:])

    return train_set, valid_set, valid_set
