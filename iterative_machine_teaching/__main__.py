import argparse

from .data import load_mnist, load_gaussian
from .train import train, TeachingType


def main() -> None:
    parser = argparse.ArgumentParser("IterativeMachineTeaching")

    parser.add_argument(
        "kind", type=TeachingType, choices=list(TeachingType)
    )

    parser.add_argument(
        "-l", "--limit-train", type=int, default=-1,
        help="Number of examples in student train dataset, negative value means max"
    )

    dataset_subparser = parser.add_subparsers(
        title="dataset",
        dest="dataset",
        required=True
    )

    mnist_parser = dataset_subparser.add_parser("mnist")
    mnist_parser.add_argument("input_pickle", type=str)

    gaussian_parser = dataset_subparser.add_parser("gaussian")
    gaussian_parser.add_argument("-c", "--centers", type=int, default=2)
    gaussian_parser.add_argument("-d", "--dim", type=int, default=8)
    gaussian_parser.add_argument(
        "-n", "--per-class-example", type=int, default=512
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        dataset = load_mnist(args.input_pickle)
    elif args.dataset == "gaussian":
        dataset = load_gaussian(args.centers, args.dim, args.per_class_example)
    else:
        raise Exception("Unrecognized dataset")

    train(
        dataset,
        args.dataset,
        args.kind,
        args.limit_train
    )


if __name__ == '__main__':
    main()
