import os
import pytest

from kpconv_torch.utils.config import valid_dataset, load_config, save_config


def test_valid_dataset():
    """
    If the dataset is supported, valid_dataset returns it, otherwise it raises a ValueError.
    """
    assert valid_dataset("S3DIS") == "S3DIS"
    with pytest.raises(ValueError):
        valid_dataset("wrong_dataset")


def test_load_config():
    assert load_config("tests/config_S3DIS.yml")["dataset"] == "S3DIS"


def test_save_config():
    config = load_config("tests/config_S3DIS.yml")
    config["input"]["test_new_item"] = "new"

    save_config("tests", config)
    assert load_config("tests/config.yml")["input"]["test_new_item"] == "new"
    os.remove("tests/config.yml")
