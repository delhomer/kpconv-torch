import argparse
from pathlib import Path

from kpconv_torch import __version__ as kpconv_version
from kpconv_torch import preprocess, test, train


def valid_dir(str_dir):
    """Build a ``pathlib.Path`` object starting from the ``str_dir`` folder."""
    path_dir = Path(str_dir)
    if not path_dir.is_dir():
        raise argparse.ArgumentTypeError(f"The {str(path_dir)} folder does not exist.")
    return path_dir


def valid_file(str_path):
    path = Path(str_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"The {str(str_path)} file does not exists")
    return path


def kpconv_parser(subparser, reference_func, command, command_description):
    """CLI definition for kpconv commands

    Parameters
    ----------
    subparser: argparser.parser.SubParsersAction
    reference_func: function
    """
    parser = subparser.add_parser(command, help=command_description)

    parser.add_argument(
        "-c",
        "--configfile",
        required=False,
        type=valid_file,
        help="Path to the config file for the chosen dataset. ",
    )

    parser.add_argument(
        "-d",
        "--datapath",
        required=True,
        type=valid_dir,
        help="Path of the dataset on the file system",
    )

    if command == "test":
        parser.add_argument(
            "-f",
            "--filename",
            required=False,
            type=valid_file,
            help=(
                "File on which to predict semantic labels starting from a trained model "
                "(if None, use the validation task)"
            ),
        )

        parser.add_argument(
            "-l",
            "--chosen-log",
            required=True,
            type=valid_dir,
            help=(
                "If mentioned with the test command, "
                "the test will use this folder for the inference procedure."
            ),
        )

    if command == "train":
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "-l",
            "--chosen-log",
            type=valid_dir,
            help=(
                "If mentioned with the train command, "
                "the training starts from an already trained model, "
                "contained in the mentioned folder."
            ),
        )

        group.add_argument(
            "-o",
            "--output-dir",
            type=valid_dir,
            help=(
                "If mentioned, starts training from the begining. "
                "Otherwise, the -l option must be mentioned."
            ),
        )

    parser.set_defaults(func=reference_func)


def main():
    """Main method of the module"""
    parser = argparse.ArgumentParser(
        prog="kpconv",
        description=(
            f"kpconv_torch version {kpconv_version}. "
            "Implementation of the Kernel Point Convolution (KPConv) algorithm with PyTorch."
        ),
    )
    sub_parsers = parser.add_subparsers(dest="command")
    kpconv_parser(
        sub_parsers,
        preprocess.main,
        "preprocess",
        "Preprocess a dataset to make it compliant with the program",
    )
    kpconv_parser(
        sub_parsers,
        train.main,
        "train",
        "Train a KPConv model",
    )
    kpconv_parser(
        sub_parsers,
        test.main,
        "test",
        "Test a KPConv trained model",
    )

    args = parser.parse_args()

    if args.config is None and args.chosen_log is None:
        raise Exception(
            "A --chosen-log / -l, with a config.yml file inside the folder "
            "or a --configfile / -c must be specified."
        )

    if "func" in vars(args):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
