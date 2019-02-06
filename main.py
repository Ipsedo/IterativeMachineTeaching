import gaussian_data
import sys

if __name__ == "__main__":
    if sys.argv[1] == "gaussian":
        gaussian_data.gaussian_main()
    elif sys.argv[1] == "mnist":
        pass
