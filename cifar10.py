import data.dataset_loader as data_loader


def cifar10_main():
    x_train,y_train = data_loader.load_cifar10_2()
    y_train = y_train.reshape(-1)

    nb_train = 1000

    class_1 = 5
    class_2 = 9

    f = (y_train == class_1) | (y_train == class_2)
    x_train, y_train = x_train[f], y_train[f]

    x_train = data_loader.cifar10_proper_array(x_train)

    x_train, y_train = x_train[:nb_train], y_train[:nb_train]

    print(x_train.shape, y_train.shape)
