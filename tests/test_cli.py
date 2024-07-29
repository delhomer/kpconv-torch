"""
CLI testing functions

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, E0401

from argparse import ArgumentTypeError
from pathlib import Path

import pytest

from kpconv_torch.cli.__main__ import valid_dataset, valid_dir


def test_valid_dataset():
    """If the dataset is supported, valid_dataset returns it, otherwise it raises an argparse
    error.

    """
    assert valid_dataset("S3DIS") == "S3DIS"
    with pytest.raises(ArgumentTypeError):
        valid_dataset("wrong_dataset")


def test_valid_dir():
    """valid_dir raises an argparse error if it is not a valid directory"""
    with pytest.raises(ArgumentTypeError):
        valid_dir(__file__)
    valid_dir(Path(__file__).parent)
