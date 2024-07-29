"""
SemanticKitti debug functions

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0902, R0913, R0912, R0914, R0915, R1702

import time

import numpy as np


def debug_timing(dataset, loader):
    """
    Timing of generator function
    """

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.batch_num
    estim_n = 0

    for _ in range(10):

        for batch_i, batch in enumerate(loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.frame_inds) - estim_b) / 100
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


def debug_class_w(dataset, loader):
    """
    Timing of generator function
    """

    i = 0

    counts = np.zeros((dataset.num_classes,), dtype=np.int64)

    s = "step"
    for c in dataset.label_names:
        s += f"{c[:4]:^6}"
    print(s)
    print(6 * "-" + "|" + 6 * dataset.num_classes * "-")

    for _ in range(10):
        for batch in loader:

            # count labels
            new_counts = np.bincount(batch.labels)

            counts[: new_counts.shape[0]] += new_counts.astype(np.int64)

            # Update proportions
            proportions = 1000 * counts / np.sum(counts)

            s = f"{i:^6d}|"
            for pp in proportions:
                s += f"{pp:^6.1f}"
            print(s)
            i += 1
