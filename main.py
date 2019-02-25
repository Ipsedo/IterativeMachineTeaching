from experiences import cifar10, mnist, gaussian_data
import sys

if __name__ == "__main__":

    if sys.argv[1] == "gaussian":
        gaussian_data.gaussian_main(sys.argv[2])
    elif sys.argv[1] == "mnist":
        mnist.mnist_main(sys.argv[2])
    elif sys.argv[1] == "cifar":
        cifar10.cifar10_main(sys.argv[2])

