"""
Load / Save configuration dictionary functions and terminal display color definition

@author: Hugues THOMAS, Oslandia
@date: july 2024
"""

from enum import Enum
import random
import yaml

from pathlib import Path


class BColors(Enum):
    """
    Colors used to display the code in the terminal
    """

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
    """
    Tells if a dataset name corresponds to a valid one

    :param dataset: dataset name
    :return: the dataset name, if it is valid, otherwise raises an Exception
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"{dataset} dataset is unknown, please choose amongst {SUPPORTED_DATASETS}."
        )
    return dataset


def save_config(train_save_path, config):
    """
    Saves a configuration into a YAML file

    :param train_save_path: a path to a folder where to save the file
    :param config: a configuration dictionnary
    """
    with open(Path(train_save_path) / "config.yml", "w", encoding="utf-8") as file_object:
        yaml.dump(config, file_object)


def load_config(file_path):
    """
    Loads a configuration from a YAML file

    :param file_path: a path to a config file
    """
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8") as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)

    if "colors" not in config:
        config["colors"] = [
            "#%06xff" % random.randint(0, 0xFFFFFF) for _ in config["model"]["label_to_names"]
        ]

    # Check if dataset exists
    valid_dataset(config["dataset"])

    return config
