import argparse

from .train import train_mnist, Kind


def main() -> None:
    parser = argparse.ArgumentParser("IterativeMachineTeaching")

    parser.add_argument(
        "kind", type=Kind, choices=list(Kind)
    )

    parser.add_argument(
        "first_class", type=int, choices=list(range(10))
    )

    parser.add_argument(
        "second_class", type=int, choices=list(range(10))
    )

    parser.add_argument(
        "input_pickle", type=str
    )

    args = parser.parse_args()

    train_mnist(
        args.input_pickle,
        args.first_class,
        args.second_class,
        args.kind
    )


if __name__ == '__main__':
    main()
