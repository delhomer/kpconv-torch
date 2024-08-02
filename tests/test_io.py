import numpy as np
from pytest import mark

from kpconv_torch.io import las, ply, xyz


@mark.dependency()
def test_write_ply_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.ply"
    res = ply.write_ply(
        str(example_filepath),
        (points_array, colors_array, classification_array),
        ["x", "y", "z", "red", "green", "blue", "classification"],
    )
    assert res and example_filepath.exists()

    read_data = ply.read_ply(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    read_classification = read_data[2]
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)
    np.testing.assert_array_equal(classification_array, read_classification)


@mark.dependency()
def test_write_ply_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.ply"
    res = ply.write_ply(
        str(example_filepath),
        [points_array, colors_array],
        ["x", "y", "z", "red", "green", "blue"],
    )
    assert res and example_filepath.exists()

    read_data = ply.read_ply(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)


@mark.dependency()
def test_write_ply_generator(fixture_path, points_array):
    example_filepath = fixture_path / "example_from_generator.ply"
    points = np.concatenate([points_array, points_array])

    def point_gen(points):
        """Produce a dummy generator for testing purpose, with two points per iteration."""
        for p in range(0, points.shape[0], 2):
            yield points[p : p + 2]

    res = ply.write_ply_from_generator(
        str(example_filepath),
        [point_gen(points)],
        ["x", "y", "z"],
        nb_points=points.shape[0],
    )
    assert res and example_filepath.exists()
    points_1, _, _ = ply.read_ply(str(example_filepath))
    assert points_1.shape == points.shape


@mark.dependency()
def test_write_las_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.las"
    res = las.write_las(str(example_filepath), points_array, colors_array, classification_array)
    assert res and example_filepath.exists()

    read_data = las.read_las_laz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    read_classification = read_data[2]
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)
    np.testing.assert_array_equal(classification_array, read_classification)


@mark.dependency()
def test_write_las_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.las"
    res = las.write_las(str(example_filepath), points_array, colors_array)
    assert res and example_filepath.exists()

    read_data = las.read_las_laz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)


@mark.dependency()
def test_write_xyz_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.xyz"
    res = xyz.write_xyz(str(example_filepath), points_array, colors_array)
    assert res and example_filepath.exists()

    read_data = xyz.read_xyz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)


@mark.dependency()
def test_write_xyz_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.xyz"
    res = xyz.write_xyz(str(example_filepath), points_array, colors_array, classification_array)
    assert res and example_filepath.exists()

    read_data = xyz.read_xyz(str(example_filepath))
    read_points = np.vstack((read_data[0][:, 0], read_data[0][:, 1], read_data[0][:, 2])).T
    read_colors = np.vstack((read_data[1][:, 0], read_data[1][:, 1], read_data[1][:, 2])).T
    read_classification = (read_data[2]).T
    np.testing.assert_array_equal(points_array, read_points)
    np.testing.assert_array_equal(colors_array, read_colors)
    np.testing.assert_array_equal(classification_array, read_classification)


@mark.dependency(depends=["test_write_xyz_without_classification"])
def test_read_xyz_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.xyz"
    points, colors, _ = xyz.read_xyz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    (fixture_path / "example_without_classification.xyz").unlink()


@mark.dependency(depends=["test_write_xyz_with_classification"])
def test_read_xyz_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.xyz"
    points, colors, classification = xyz.read_xyz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    np.testing.assert_array_equal(classification_array, classification)
    (fixture_path / "example_with_classification.xyz").unlink()


@mark.dependency(depends=["test_write_ply_without_classification"])
def test_read_ply_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.ply"
    points, colors, _ = ply.read_ply(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    (fixture_path / "example_without_classification.ply").unlink()


@mark.dependency(depends=["test_write_ply_with_classification"])
def test_read_ply_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.ply"
    points, colors, classification = ply.read_ply(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    np.testing.assert_array_equal(classification_array, classification)
    (fixture_path / "example_with_classification.ply").unlink()


@mark.dependency(depends=["test_write_las_without_classification"])
def test_read_las_without_classification(fixture_path, points_array, colors_array):
    example_filepath = fixture_path / "example_without_classification.las"
    points, colors, _ = las.read_las_laz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    (fixture_path / "example_without_classification.las").unlink()


@mark.dependency(depends=["test_write_las_with_classification"])
def test_read_las_with_classification(
    fixture_path, points_array, colors_array, classification_array
):
    example_filepath = fixture_path / "example_with_classification.las"
    points, colors, classification = las.read_las_laz(example_filepath)
    np.testing.assert_array_equal(points_array, points)
    np.testing.assert_array_equal(colors_array, colors)
    np.testing.assert_array_equal(classification_array, classification)
    (fixture_path / "example_with_classification.las").unlink()
