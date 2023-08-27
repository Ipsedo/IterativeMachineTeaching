# -*- coding: utf-8 -*-
import argparse

from .data import load_gaussian, load_mnist
from .train import (
    DatasetOptions,
    StudentOptions,
    TeacherOptions,
    TeachingType,
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser("IterativeMachineTeaching")

    parser.add_argument(
        "--cuda",
        action="store_true",
    )
    parser.add_argument(
        "kind",
        type=TeachingType,
        choices=list(TeachingType),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=4.0 / 5.0,
    )

    # student / example options
    parser.add_argument(
        "--student-examples",
        type=int,
        default=-1,
        help="Number of examples in student train dataset, "
        "negative value means max",
    )
    parser.add_argument(
        "--student-steps",
        type=int,
        default=1024,
        help="Number of forward / backward steps for student",
    )
    parser.add_argument(
        "--student-batch-size",
        type=int,
        default=8,
        help="Batch size for student and example",
    )
    parser.add_argument(
        "--student-lr",
        type=float,
        default=1e-3,
        help="Student and example learning rate",
    )

    # teacher options
    parser.add_argument(
        "--teacher-lr",
        type=float,
        default=1e-3,
        help="Teacher learning rate",
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=8,
        help="Teacher batch size",
    )
    parser.add_argument(
        "--research-batch-size",
        type=int,
        default=512,
        help="Batch size for example research",
    )
    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=16,
        help="Teacher training epochs",
    )

    dataset_subparser = parser.add_subparsers(
        title="dataset",
        dest="dataset",
        required=True,
    )

    mnist_parser = dataset_subparser.add_parser("mnist")
    mnist_parser.add_argument(
        "input_pickle",
        type=str,
    )

    gaussian_parser = dataset_subparser.add_parser("gaussian")
    gaussian_parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=8,
    )
    gaussian_parser.add_argument(
        "-n",
        "--per-class-example",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        x, y = load_mnist(args.input_pickle)
    elif args.dataset == "gaussian":
        x, y = load_gaussian(args.dim, args.per_class_example)
    else:
        parser.error("Unrecognized dataset")
        return

    dataset_options = DatasetOptions(args.dataset, x, y, args.train_ratio)

    student_options = StudentOptions(
        args.student_examples,
        args.student_steps,
        args.student_batch_size,
        args.student_lr,
    )

    teacher_options = TeacherOptions(
        args.teacher_lr,
        args.teacher_batch_size,
        args.research_batch_size,
        args.teacher_epochs,
    )

    train(
        dataset_options,
        args.kind,
        teacher_options,
        student_options,
        args.cuda,
    )


if __name__ == "__main__":
    main()
