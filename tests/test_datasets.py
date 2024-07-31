"""
Unit tests related to the datasets classes
"""

import numpy as np

from kpconv_torch.datasets.common import PointCloudDataset
from kpconv_torch.utils.config import load_config


def test_augmentation_transform(input_points_array, dataset_path):
    """
    Unit test for augmentation_transform
    """
    # To make the result predictable
    np.random.seed(0)

    # Load configuration
    config = load_config("tests/tests_config_S3DIS.yml")

    dataset = PointCloudDataset(
        config,
        dataset_path,
        ignored_labels=np.array([]),
        chosen_log=None,
        infered_file=None,
        task="train",
    )
    augmented_points, scale, var_r = dataset.augmentation_transform(input_points_array)

    assert np.all(
        (augmented_points * 100).astype(np.int32)
        == [
            [-83, 86, 20],
            [-85, 82, 20],
            [-88, 81, 20],
            [11, 17, 116],
            [12, 17, 115],
            [12, 17, 114],
        ]
    )

    assert np.all((scale * 100).astype(np.int32) == [-104, 102, 100])

    assert np.all(
        (var_r * 100).astype(np.int32)
        == [
            [-95, 30, 0],
            [-30, -95, 0],
            [0, 0, 100],
        ]
    )
