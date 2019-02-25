import gaussian_data
import mnist
import cifar10
import sys
# import data.dataset_loader as dataset_loader

if __name__ == "__main__":

    if sys.argv[1] == "gaussian":
        gaussian_data.gaussian_main(sys.argv[2])
    elif sys.argv[1] == "mnist":
        mnist.mnist_main(sys.argv[2])
    elif sys.argv[1] == "cifar":
        cifar10.cifar10_main(sys.argv[2])

