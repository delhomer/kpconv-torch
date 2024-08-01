from argparse import ArgumentTypeError
from pathlib import Path

import pytest

from kpconv_torch.cli.__main__ import valid_dir


def test_valid_dir():
    """valid_dir raises an argparse error if it is not a valid directory"""
    with pytest.raises(ArgumentTypeError):
        valid_dir(__file__)
    valid_dir(Path(__file__).parent)
