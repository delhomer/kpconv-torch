"""
Mother Dataset class

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

import numpy as np
from torch.utils.data import Dataset

from kpconv_torch.io.io import get_test_save_path
from kpconv_torch.kernels.kernel_points import create_3d_rotations
from kpconv_torch.utils.visu_mayavi_functions import show_modelnet_examples
from kpconv_torch.datasets.common_functions import batch_grid_subsampling, batch_neighbors

# pylint: disable=R0913, R0914, R0912, R0902, R0915, E0401, C0103


class PointCloudDataset(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(
        self,
        config,
        datapath,
        ignored_labels,
        chosen_log=None,
        infered_file=None,
        task="train",
    ):
        """
        Initialize parameters of the dataset here.
        """
        self.config = config
        self.ignored_labels = ignored_labels
        self.label_values = np.zeros((0,), dtype=np.int32)
        self.label_names = []
        self.label_to_idx = {}
        self.name_to_label = {}
        self.neighborhood_limits = []

        # Training or test set
        if task not in ["train", "validate", "test", "ERF", "all"]:
            raise ValueError("Unknown task for the dataset: ", task)

        self.task = task

        self.datapath = datapath

        self.test_save_path = get_test_save_path(infered_file, chosen_log)

        # Number of layers
        self.num_layers = (
            len(
                [
                    block
                    for block in config["model"]["architecture"]
                    if "pool" in block or "strided" in block
                ]
            )
            + 1
        )

        # Deform layer list
        # List of boolean indicating which layer has a deformable convolution

        layer_blocks = []
        self.deform_layers = []
        for block in config["model"]["architecture"]:

            # Get all blocks of the layer
            if not (
                "pool" in block or "strided" in block or "global" in block or "upsample" in block
            ):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            deform_layer = False
            if layer_blocks and np.any(["deformable" in blck for blck in layer_blocks]):
                deform_layer = True

            if ("pool" in block or "strided" in block) and "deformable" in block:
                deform_layer = True

            self.deform_layers += [deform_layer]
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if "global" in block or "upsample" in block:
                break

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.config["model"]["label_to_names"])
        self.label_values = np.sort([k for k, _ in self.config["model"]["label_to_names"].items()])
        self.label_names = [self.config["model"]["label_to_names"][k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.config["model"]["label_to_names"].items()}

    def __len__(self):
        """
        Return the length of data here
        """
        return 0

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """

        return 0

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        # Rotation
        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config["train"]["augment_rotation"] == "vertical":

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config["train"]["augment_rotation"] == "all":

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array(
                    [
                        np.cos(theta) * np.cos(phi),
                        np.sin(theta) * np.cos(phi),
                        np.sin(phi),
                    ]
                )

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3d_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        # Scale
        # Choose random scales for each example
        min_s = self.config["train"]["augment_scale_min"]
        max_s = self.config["train"]["augment_scale_max"]
        if self.config["train"]["augment_scale_anisotropic"]:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) + min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config["train"]["augment_symmetries"]).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        # Noise
        noise = (
            np.random.randn(points.shape[0], points.shape[1])
            * self.config["train"]["augment_noise"]
        ).astype(np.float32)

        # Apply transforms
        # Do not use np.dot because it is multi-threaded
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise

        if normals is None:
            return augmented_points, scale, R

        # Anisotropic scale of the normals thanks to cross product formula
        normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
        augmented_normals = np.dot(normals, R) * normal_scale
        # Renormalise
        augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

        if verbose:
            test_p = [np.vstack([points, augmented_points])]
            test_n = [np.vstack([normals, augmented_normals])]
            test_l = [np.hstack([points[:, 2] * 0, augmented_points[:, 2] * 0 + 1])]
            show_modelnet_examples(test_p, test_n, test_l)

        return augmented_points, augmented_normals, scale, R

    def big_neighborhood_filter(self, neighbors, layer):
        """Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the
        neighborhoods untouched. Limit is computed at initialization.

        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, : self.neighborhood_limits[layer]]

        return neighbors

    def classification_inputs(self, stacked_points, stacked_features, labels, stack_lengths):
        """
        docstring to do
        """
        # Starting radius of convolutions
        r_normal = (
            self.config["kpconv"]["first_subsampling_dl"] * self.config["kpconv"]["conv_radius"]
        )

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        deform_layers = []

        # Loop over the blocks
        for block in self.config["model"]["architecture"]:
            # Get all blocks of the layer
            if not (
                "pool" in block or "strided" in block or "global" in block or "upsample" in block
            ):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(["deformable" in blck for blck in layer_blocks]):
                    r = (
                        r_normal
                        * self.config["train"]["batch_num"]
                        / self.config["kpconv"]["conv_radius"]
                    )
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(
                    stacked_points, stacked_points, stack_lengths, stack_lengths, r
                )

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # If end of layer is a pooling operation
            if "pool" in block or "strided" in block:

                # New subsampling length
                dl = 2 * r_normal / self.config["kpconv"]["conv_radius"]

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if "deformable" in block:
                    r = (
                        r_normal
                        * self.config["train"]["batch_num"]
                        / self.config["kpconv"]["conv_radius"]
                    )
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if "global" in block or "upsample" in block:
                break

        # Return inputs
        # Save deform layers

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_stack_lengths
        li += [stacked_features, labels]

        return li

    def segmentation_inputs(self, stacked_points, stacked_features, labels, stack_lengths):
        """
        docstring to do
        """
        # Starting radius of convolutions
        r_normal = (
            self.config["kpconv"]["first_subsampling_dl"] * self.config["kpconv"]["conv_radius"]
        )

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        # Loop over the blocks
        for block in self.config["model"]["architecture"]:
            # Get all blocks of the layer
            if not (
                "pool" in block or "strided" in block or "global" in block or "upsample" in block
            ):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer
                # compute the neighbors with the good radius
                if np.any(["deformable" in blck for blck in layer_blocks]):
                    r = (
                        r_normal
                        * self.config["kpconv"]["deform_radius"]
                        / self.config["kpconv"]["conv_radius"]
                    )
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(
                    stacked_points, stacked_points, stack_lengths, stack_lengths, r
                )

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # If end of layer is a pooling operation
            if "pool" in block or "strided" in block:

                # New subsampling length
                dl = 2 * r_normal / self.config["kpconv"]["conv_radius"]

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if "deformable" in block:
                    r = (
                        r_normal
                        * self.config["train"]["batch_num"]
                        / self.config["kpconv"]["conv_radius"]
                    )
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points) + 1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if "global" in block or "upsample" in block:
                break

        # Return inputs
        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths
        li += [stacked_features, labels]

        return li
