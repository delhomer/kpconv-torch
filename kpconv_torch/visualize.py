"""
Models visualizing functions

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

from pathlib import Path
import os
import time

from torch.utils.data import DataLoader
import numpy as np

from kpconv_torch.datasets.modelnet40 import (
    modelnet40_collate,
    ModelNet40Dataset,
    ModelNet40Sampler,
)
from kpconv_torch.datasets.s3dis import (
    s3dis_collate,
    S3DISDataset,
    S3DISSampler,
)
from kpconv_torch.models.architectures import KPCNN, KPFCNN
from kpconv_torch.utils.config import load_config
from kpconv_torch.utils.visualizer import ModelVisualizer


def main(args):
    """
    Launch the visualization from the CLI arguments
    """
    visualize(args.dataset, args.datapath, args.chosen_log)


def visualize(dataset: str, datapath: Path, chosen_log: Path) -> None:
    """
    Visualization of the training results
    """

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Eventually you can choose which feature is visualized (index of the deform convolution in the
    # network)
    deform_idx = 0

    # Initialize the environment
    # Set which gpu is going to be used
    gpu_id = "0"

    # Set GPU visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Previous chkp
    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, "checkpoints")
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == "chkp"]

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = "current_chkp.tar"
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, "checkpoints", chosen_chkp)

    # Initialize configuration class
    config = load_config(chosen_log)

    # Change model parameters for test
    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config["train"]["augment_noise"] = 0.0001
    config["train"]["batch_num"] = 1
    config["train"]["sphere_radius"] = 2.0
    config["input"]["threads"] = 0

    # Prepare Data
    print()
    print("Data Preparation")
    print("****************")

    # Initiate dataset
    if dataset == "ModelNet40":
        test_dataset = ModelNet40Dataset(config=config, datapath=datapath, task="validate")
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = modelnet40_collate
    elif dataset == "S3DIS":
        test_dataset = S3DISDataset(config=config, datapath=datapath, task="validate")
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = s3dis_collate
    else:
        raise ValueError("Unsupported dataset : " + config["model"]["dataset"])

    # Data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config.threads,
        pin_memory=True,
    )

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print("\nModel Preparation")
    print("*******************")

    # Define network model
    t1 = time.time()
    if config["model"]["task"] == "classification":
        net = KPCNN(config)
    elif config["model"]["task"] in ["cloud_segmentation", "slam_segmentation"]:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError("Unsupported task for deformation visu: " + config["model"]["task"])

    # Define a visualizer class
    visualizer = ModelVisualizer(net, config, chkp_path=chosen_chkp, on_gpu=False)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart visualization")
    print("*********************")

    # Visualizing
    visualizer.show_deformable_kernels(net, test_loader, deform_idx)
