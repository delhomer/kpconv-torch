from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest


@pytest.fixture
def fixture_path():
    yield Path(__file__).parent / "fixtures"


@pytest.fixture
def dataset_path(fixture_path):
    yield fixture_path / "S3DIS"


@pytest.fixture
def trained_model_path(fixture_path):
    yield fixture_path / "trained_models"


@pytest.fixture
def training_log(trained_model_path):
    chosen_log_dir = next(trained_model_path.iterdir())
    yield chosen_log_dir


@pytest.fixture
def inference_file(fixture_path, training_log):
    yield fixture_path / "inference" / "Area4_hallway5.ply"
    rmtree(training_log)
    rmtree(fixture_path / "S3DIS" / "calibration")
    rmtree(fixture_path / "S3DIS" / "input_0.030")
    rmtree(fixture_path / "S3DIS" / "original_ply")
    for p in Path(fixture_path / "inference").glob("Area4_hallway5.ply_*"):
        p.unlink()
    for p in Path(fixture_path / "inference").glob("Area4_hallway5.ply.*"):
        p.unlink()


@pytest.fixture
def points_array():
    yield np.array(
        [
            [5.83, -18.83, 0.24],
            [5.83, -18.86, 0.22],
            [5.82, -18.84, 0.24],
            [5.83, -18.84, 0.23],
            [5.82, -18.88, 0.23],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def colors_array():
    yield np.array(
        [
            [158, 167, 176],
            [151, 162, 176],
            [157, 164, 176],
            [155, 165, 176],
            [154, 167, 177],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def classification_array():
    yield np.array(
        [1, 2, 2, 3, 1],
        dtype=np.int32,
    )


@pytest.fixture
def input_points_array(fixture_path):
    yield np.array(
        [
            [-0.509, -1.047, 0.205],
            [-0.539, -1.018, 0.207],
            [-0.566, -1.021, 0.206],
            [0.157, -0.127, 1.152],
            [0.165, -0.128, 1.145],
            [0.166, -0.128, 1.136],
        ],
        dtype=np.float32,
    )
