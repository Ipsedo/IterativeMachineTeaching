# -*- coding: utf-8 -*-
import sys

from .experiences import cifar10_main, gaussian_main, mnist_main


def main() -> None:

    if sys.argv[1] == "gaussian":
        gaussian_main(sys.argv[2])
    elif sys.argv[1] == "mnist":
        mnist_main(sys.argv[2])
    elif sys.argv[1] == "cifar":
        cifar10_main(sys.argv[2])


if __name__ == "__main__":
    main()
