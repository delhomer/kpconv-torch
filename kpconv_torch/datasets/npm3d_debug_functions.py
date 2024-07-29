"""
NPM3D Dataset Class, used to manage data that can be downloaded here :
https://npm3d.fr/

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915

import time

import numpy as np
from kpconv_torch.utils.visu_mayavi_functions import show_input_batch


def debug_upsampling(dataset, loader):
    """
    Shows which labels are sampled according to strategy chosen
    """

    for _ in range(10):

        for batch in loader:
            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, "=>", pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0**2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print("******************")
        print("*******************************************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """
    Timing of generator function
    """

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config["train"]["batch_num"]
    estim_n = 0

    for _ in range(10):

        for batch_i, batch in enumerate(loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_n += (batch.features.shape[0] - estim_n) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = "Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}"
                print(
                    message.format(batch_i, 1000 * mean_dt[0], 1000 * mean_dt[1], estim_b, estim_n)
                )

        print("************* Epoch ended *************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, config, loader):
    """
    :param dataset
    :param config
    :param loader
    """
    for _ in range(10):
        layers = config["model"]["num_layers"]

        for batch in loader:

            # Print characteristics of input tensors
            print("\nPoints tensors")
            for i in range(layers):
                print(batch.points[i].dtype, batch.points[i].shape)
            print("\nNeighbors tensors")
            for i in range(layers):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print("\nPools tensors")
            for i in range(layers):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print("\nStack lengths")
            for i in range(layers):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print("\nFeatures")
            print(batch.features.dtype, batch.features.shape)
            print("\nLabels")
            print(batch.labels.dtype, batch.labels.shape)
            print("\nAugment Scales")
            print(batch.scales.dtype, batch.scales.shape)
            print("\nAugment Rotations")
            print(batch.rots.dtype, batch.rots.shape)
            print("\nModel indices")
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print("\nAre input tensors pinned")
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print("*******************************************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """
    Timing of generator function
    """

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for _ in range(10):

        for batch_i, _ in enumerate(loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = "Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} "
                print(message.format(batch_i, 1000 * mean_dt[0], 1000 * mean_dt[1]))

        print("************* Epoch ended *************")

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)
