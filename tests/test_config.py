import pytest

from kpconv_torch.utils.config import valid_dataset


def test_valid_dataset():
    """If the dataset is supported, valid_dataset returns it, otherwise it raises an argparse
    error.

    """
    assert valid_dataset("S3DIS") == "S3DIS"
    with pytest.raises(ValueError):
        valid_dataset("wrong_dataset")
