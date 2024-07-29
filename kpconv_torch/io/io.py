"""
Path definition functions for I/O

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, C0103, E0401

import os
import time
from pathlib import Path


def get_train_save_path(output_dir: Path, chosen_log: Path) -> Path:
    """
    Function that returns the path where to save the model training results

    :param output_dir: CLI parameter -o
    :param chosen_log: folder containing an already trained model
    :returns path where to save the model training results
    """
    if chosen_log is None and output_dir is None:
        train_path = None
    elif chosen_log is not None:
        train_path = chosen_log
    elif output_dir is not None:
        train_path = output_dir / time.strftime("Log_%Y-%m-%d_%H-%M-%S", time.gmtime())
    if train_path is not None and not os.path.exists(train_path):
        os.makedirs(train_path)
    return train_path


def get_test_save_path(infered_file: Path, chosen_log: Path) -> Path:
    if chosen_log is None:
        test_path = None
    elif infered_file is not None:
        test_path = Path(infered_file).parent / "test" / Path(chosen_log).name
    else:
        test_path = Path(chosen_log) / "test"
    if test_path is not None and not os.path.exists(test_path):
        os.makedirs(test_path)
    return test_path
