import numpy as np
from pytest import mark

from kpconv_torch import preprocess
from kpconv_torch.io import ply
from kpconv_torch.datasets.common import PointCloudDataset
from kpconv_torch.utils.config import load_config


@mark.dependency()
def test_preprocess(dataset_path):
    """The preprocessing produces .PLY files usable for further training process in 'original_ply'
    subfolder as well as subsampling material in 'input_{subsampling_coef}' subfolder (KDTree,
    coarse KDTree and subsampled .PLY file).

    Expected output tree:

    _ tests/fixtures/S3DIS/
        |_ input_0.030/
            |_ Area_3_coarse_KDTree.pkl
            |_ Area_3_KDTree.pkl
            |_ Area_3.ply
            |_ Area_5_coarse_KDTree.pkl
            |_ Area_5_KDTree.pkl
            |_ Area_5.ply
        |_ original_ply/
            |_ Area_3.ply
            |_ Area_5.ply
    """
    preprocess.preprocess(dataset_path, "tests/tests_config_S3DIS.yml")
    subsampling_coef = 0.03
    assert (dataset_path / f"input_{subsampling_coef:.3f}").exists()
    assert (dataset_path / "original_ply").exists()
    for room_name, cloud_name in zip(("hallway_3", "storage_3"), ("Area_3", "Area_5")):
        # Check the output tree
        assert (
            dataset_path / f"input_{subsampling_coef:.3f}" / f"{cloud_name}_coarse_KDTree.pkl"
        ).exists()
        assert (
            dataset_path / f"input_{subsampling_coef:.3f}" / f"{cloud_name}_KDTree.pkl"
        ).exists()
        assert (dataset_path / f"input_{subsampling_coef:.3f}" / f"{cloud_name}.ply").exists()
        assert (dataset_path / "original_ply" / f"{cloud_name}.ply").exists()

        # Check the output PLY content
        raw_data = np.loadtxt(dataset_path / cloud_name / room_name / f"{room_name}.txt")
        points, colors, labels = ply.read_ply(dataset_path / "original_ply" / f"{cloud_name}.ply")
        assert points.shape == (raw_data.shape[0], 3)
        assert colors.shape == (raw_data.shape[0], 3)
        assert labels.shape == (raw_data.shape[0],)

        expected_label_count = len(
            {
                filename.name.split("_")[0]
                for filename in (dataset_path / cloud_name / room_name / "Annotations").iterdir()
            }
        )
        assert len(np.unique(labels)) == expected_label_count


@mark.dependency(depends=["test_preprocess"])
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
