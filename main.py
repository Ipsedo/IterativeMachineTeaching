import gaussian_data
import mnist
import cifar10
import sys
# import data.dataset_loader as dataset_loader

if __name__ == "__main__":

    if sys.argv[1] == "gaussian":
        if sys.argv[2] == "omni":
            gaussian_data.gaussian_omniscient_main()
        elif sys.argv[2] == "surro":
            gaussian_data.gaussian_surrogate_main()
    elif sys.argv[1] == "mnist":
        if sys.argv[2] == "omni":
            mnist.mnist_data_omniscient()
        elif sys.argv[2] == "surro":
            mnist.mnist_data_surrogate()
    elif sys.argv[1] == "cifar":
        cifar10.cifar10_main()

