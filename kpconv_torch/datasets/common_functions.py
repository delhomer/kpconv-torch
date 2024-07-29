"""
All datasets common functions

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, E0401, C0103

import numpy as np

import radius_neighbors as cpp_neighbors
import grid_subsampling as cpp_subsampling
from kpconv_torch.kernels.kernel_points import create_3d_rotations


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points, sampleDl=sampleDl, verbose=verbose)

    if labels is None:
        return cpp_subsampling.subsample(
            points, features=features, sampleDl=sampleDl, verbose=verbose
        )

    if features is None:
        return cpp_subsampling.subsample(points, classes=labels, sampleDl=sampleDl, verbose=verbose)

    return cpp_subsampling.subsample(
        points,
        features=features,
        classes=labels,
        sampleDl=sampleDl,
        verbose=verbose,
    )


def batch_grid_subsampling(
    points,
    batches_len,
    features=None,
    labels=None,
    sampleDl=0.1,
    max_p=0,
    verbose=0,
    random_grid_orient=True,
):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        # Create a random rotation matrix for each batch element
        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3d_rotations(u.T, alpha).astype(np.float32)

        # Apply rotations
        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0 : i0 + length, :] = np.sum(
                np.expand_dims(points[i0 : i0 + length, :], 2) * R[bi], axis=1
            )
            i0 += length

    # Sunsample and realign
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(
            points, batches_len, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0 : i0 + length, :] = np.sum(
                    np.expand_dims(s_points[i0 : i0 + length, :], 2) * R[bi].T, axis=1
                )
                i0 += length
        return s_points, s_len

    if labels is None:
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(
            points,
            batches_len,
            features=features,
            sampleDl=sampleDl,
            max_p=max_p,
            verbose=verbose,
        )
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0 : i0 + length, :] = np.sum(
                    np.expand_dims(s_points[i0 : i0 + length, :], 2) * R[bi].T, axis=1
                )
                i0 += length
        return s_points, s_len, s_features

    if features is None:
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(
            points,
            batches_len,
            classes=labels,
            sampleDl=sampleDl,
            max_p=max_p,
            verbose=verbose,
        )
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0 : i0 + length, :] = np.sum(
                    np.expand_dims(s_points[i0 : i0 + length, :], 2) * R[bi].T, axis=1
                )
                i0 += length
        return s_points, s_len, s_labels

    s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(
        points,
        batches_len,
        features=features,
        classes=labels,
        sampleDl=sampleDl,
        max_p=max_p,
        verbose=verbose,
    )
    if random_grid_orient:
        i0 = 0
        for bi, length in enumerate(s_len):
            # Apply the rotation
            s_points[i0 : i0 + length, :] = np.sum(
                np.expand_dims(s_points[i0 : i0 + length, :], 2) * R[bi].T, axis=1
            )
            i0 += length
    return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
