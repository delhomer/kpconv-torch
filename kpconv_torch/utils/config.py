from enum import Enum
import random
import yaml

from pathlib import Path


class BColors(Enum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


SUPPORTED_DATASETS = {"ModelNet40", "NPM3D", "S3DIS", "SemanticKitti", "Toronto3D"}


def valid_dataset(dataset):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"{dataset} dataset is unknown, please choose amongst {SUPPORTED_DATASETS}."
        )
    return dataset


def save_config(train_save_path, config):
    with open(Path(train_save_path) / "config.yml", "w") as file_object:
        yaml.dump(config, file_object)


def load_config(file_path):
    file_path = Path(file_path)

    with open(file_path) as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)

    if "colors" not in config:
        config["colors"] = [
            "#%06xff" % random.randint(0, 0xFFFFFF) for _ in config["model"]["label_to_names"]
        ]

    # Check if dataset exists
    valid_dataset(config["dataset"])

    return config
