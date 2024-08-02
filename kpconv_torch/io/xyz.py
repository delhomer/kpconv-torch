import numpy as np


def read_xyz(filepath, xyz_only=False):
    """Takes a file path pointing on a 3D point .xyz (text) file and returns the points,
    the associated colors and the associated classes.

    :param filepath: path to a 3D points file with .xyz
    :type filepath: str

    :returns: 2D np.array with type float32, 2D np.array with type uint8, 1D np.array with type
    int32
    :rtype: tuple

    """
    data = np.loadtxt(filepath, delimiter=" ")
    points = data[:, :3].astype(np.float32)
    if xyz_only:
        return points, None, None
    if data.shape[1] >= 6:
        colors = data[:, 3:6].astype(np.uint8)
    else:
        colors = colors.shape[0] > 0
    if data.shape[1] == 4:
        labels = np.squeeze(data[:, 3]).astype(np.int8)
    if data.shape[1] == 7:
        labels = np.squeeze(data[:, 6]).astype(np.int8)

    return points, colors, labels


def write_xyz(filepath, points, colors=None, labels=None):
    """Creates a .xyz file from a 3D point cloud.

    :param filepath: path to the .las file
    :type filepath: str
    :param points: 2D np.array with type float32
    :type points: np.array
    :param colors: 2D np.array with type uint8 or uint16
    :type colors: np.array
    :param labels: 1D np.array with type int32
    :type labels: np.array

    :returns: True if the file is correctly written
    :rtype: boolean

    """
    if colors is None:
        colors = np.zeros(points.shape[0], 3).astype(np.uint8)
    if labels is None:
        labels = np.zeros(points.shape[0]).astype(np.int32)
    data = np.column_stack((np.column_stack((points, colors)), labels))
    np.savetxt(filepath, data, delimiter=" ")

    return True
