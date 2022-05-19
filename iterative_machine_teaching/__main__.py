import argparse

from .data import load_mnist, load_gaussian
from .train import train, Kind


def main() -> None:
    parser = argparse.ArgumentParser("IterativeMachineTeaching")

    parser.add_argument(
        "kind", type=Kind, choices=list(Kind)
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
    gaussian_parser.add_argument("dim", type=int)
    gaussian_parser.add_argument(
        "-n", "--nb-example", type=int, default=2000
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        dataset = load_mnist(args.input_pickle)
    elif args.dataset == "gaussian":
        dataset = load_gaussian(args.dim, args.nb_example // 2)
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
