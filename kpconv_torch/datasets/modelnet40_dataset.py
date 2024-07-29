"""
ModelNet40 Dataset Class, used to manage data that can be downloaded here :
https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, E0401

import os
import pickle
import time

import numpy as np

from kpconv_torch.datasets.common_functions import grid_subsampling
from kpconv_torch.datasets.pointcloud_dataset import PointCloudDataset


class ModelNet40Dataset(PointCloudDataset):
    """
    Class to handle ModelNet40 dataset.
    """

    def __init__(
        self,
        config,
        datapath,
        chosen_log=None,
        infered_file=None,
        orient_correction=True,
        task="train",
    ):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        super().__init__(
            config=config,
            datapath=datapath,
            ignored_labels=np.array([]),
            chosen_log=chosen_log,
            infered_file=infered_file,
            task=task,
        )
        # Number of models and models used per epoch
        if self.task == "train":
            self.num_models = 9843
            if (
                self.config["train"]["epoch_steps"]
                and self.config["train"]["epoch_steps"] * self.config["train"]["batch_num"]
                < self.num_models
            ):
                self.epoch_n = (
                    self.config["train"]["epoch_steps"] * self.config["train"]["batch_num"]
                )
            else:
                self.epoch_n = self.num_models
        else:
            self.num_models = 2468
            self.epoch_n = min(
                self.num_models,
                self.config["train"]["validation_size"] * self.config["train"]["batch_num"],
            )

        # Load models
        if 0 < self.config["kpconv"]["first_subsampling_dl"] <= 0.01:
            raise ValueError("subsampling_parameter too low (should be over 1 cm")

        (
            self.input_points,
            self.input_normals,
            self.input_labels,
        ) = self.load_subsampled_clouds(orient_correction)

    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_models

    def __getitem__(self, idx_list):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in
        parallel to load a different list of indices.
        """

        # Gather batch data
        tp_list = []
        tn_list = []
        tl_list = []
        ti_list = []
        s_list = []
        r_list = []

        for p_i in idx_list:
            # Get points and labels
            points = self.input_points[p_i].astype(np.float32)
            normals = self.input_normals[p_i].astype(np.float32)
            label = self.label_to_idx[self.input_labels[p_i]]

            # Data augmentation
            points, normals, scale, var_r = self.augmentation_transform(points, normals)

            # Stack batch
            tp_list += [points]
            tn_list += [normals]
            tl_list += [label]
            ti_list += [p_i]
            s_list += [scale]
            r_list += [var_r]

        # Concatenate batch
        stacked_points = np.concatenate(tp_list, axis=0)
        stacked_normals = np.concatenate(tn_list, axis=0)
        labels = np.array(tl_list, dtype=np.int64)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(r_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config["input"]["features_dim"] == 1:
            pass
        elif self.config["input"]["features_dim"] == 4:
            stacked_features = np.hstack((stacked_features, stacked_normals))
        else:
            raise ValueError("Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)")

        # Create network inputs
        #   Points, neighbors, pooling indices for each layers
        # Get the whole input list
        input_list = self.classification_inputs(
            stacked_points, stacked_features, labels, stack_lengths
        )

        # Add scale and rotation for testing
        input_list += [scales, rots, model_inds]

        return input_list

    def load_subsampled_clouds(self, orient_correction):
        """
        :param orient_correction
        """

        # Restart timer
        var_t0 = time.time()

        # Load wanted points if possible
        if self.task == "train":
            task = "train"
        else:
            task = "test"

        var_t = self.config["kpconv"]["first_subsampling_dl"]
        print(f"\nLoading {task} points subsampled at {var_t:3f}")
        filename = os.path.join(self.datapath, f"{task}_{var_t:3f}_record.pkl")

        if os.path.exists(filename):
            with open(filename, "rb") as file:
                input_points, input_normals, input_labels = pickle.load(file)

        # Else compute them from original points
        else:

            # Collect train file names
            if self.task == "train":
                names = np.loadtxt(
                    os.path.join(self.datapath, "modelnet40_train.txt"), dtype=np.str
                )
            else:
                names = np.loadtxt(os.path.join(self.datapath, "modelnet40_test.txt"), dtype=np.str)

            # Initialize containers
            input_points = []
            input_normals = []

            # Advanced display
            var_n = len(names)
            progress_n = 30
            fmt_str = "[{:<" + str(progress_n) + "}] {:5.1f}%"

            # Collect point clouds
            for var_i, cloud_name in enumerate(names):

                # Read points
                class_folder = "_".join(cloud_name.split("_")[:-1])
                txt_file = os.path.join(self.datapath, class_folder, cloud_name) + ".txt"
                data = np.loadtxt(txt_file, delimiter=",", dtype=np.float32)

                # Subsample them
                if self.config["kpconv"]["first_subsampling_dl"] > 0:
                    points, normals = grid_subsampling(
                        data[:, :3],
                        features=data[:, 3:],
                        sampledl=self.config["kpconv"]["first_subsampling_dl"],
                    )
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                print("", end="\r")
                print(
                    fmt_str.format("#" * ((var_i * progress_n) // var_n), 100 * var_i / var_n),
                    end="",
                    flush=True,
                )

                # Add to list
                input_points += [points]
                input_normals += [normals]

            print("", end="\r")
            print(fmt_str.format("#" * progress_n, 100), end="", flush=True)
            print()

            # Get labels
            label_names = ["_".join(name.split("_")[:-1]) for name in names]
            input_labels = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
            with open(filename, "wb") as file:
                pickle.dump((input_points, input_normals, input_labels), file)

        lengths = [p.shape[0] for p in input_points]
        sizes = [length * 4 * 6 for length in lengths]
        print(f"{np.sum(sizes) * 1e-6:.1f} MB loaded in {time.time() - var_t0:.1f}s")

        if orient_correction:
            input_points = [pp[:, [0, 2, 1]] for pp in input_points]
            input_normals = [nn[:, [0, 2, 1]] for nn in input_normals]

        return input_points, input_normals, input_labels
