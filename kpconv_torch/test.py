"""
Models testing functions

@author: Hugues THOMAS, Oslandia
@date: july 2024
"""

from pathlib import Path
import os
import time

import numpy as np
from torch.utils.data import DataLoader

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
from kpconv_torch.datasets.semantickitti import (
    semantickitti_collate,
    SemanticKittiDataset,
    SemanticKittiSampler,
)
from kpconv_torch.datasets.toronto3d import (
    toronto_3d_collate,
    Toronto3DDataset,
    Toronto3DSampler,
)
from kpconv_torch.models.architectures import KPCNN, KPFCNN
from kpconv_torch.utils.config import load_config
from kpconv_torch.utils.tester import ModelTester, get_test_save_path


def main(args):
    """
    Launch the testing from the CLI arguments
    """
    test(
        args.datapath,
        args.configfile,
        args.filename,
        args.chosen_log,
    )


def test(
    datapath: Path,
    configfile: Path,
    filename: str,
    chosen_log: Path,
) -> None:
    """
    Model testing
    :param datapath: path to the data folder
    :param configfile: path to the config file
    :param filename: path to the file to use to apply the inference
    :param chosen_log: path to an already trained model which will be used to infer values
    """

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Deal with 'last_XXXXXX' choices
    output_path = get_test_save_path(filename, chosen_log)
    if configfile is not None:
        config_file_path = configfile
    else:
        config_file_path = Path(chosen_log / "config.yml")
    config = load_config(config_file_path)

    if "validation_size" not in config["test"]:
        # consider a default value if not specified
        config["test"]["validation_size"] = 200
    config["input"]["threads"] = 10

    # Initialize the environment
    # Set which gpu is going to be used
    gpu_id = "0"

    # Set GPU visible device
    os.environ["cuda_visible_devices"] = gpu_id

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

    # Prepare Data
    print()
    print("Data Preparation")
    print("****************")

    task = "validate" if filename is None else "test"

    # Initiate dataset
    if config["dataset"] == "ModelNet40":

        test_dataset = ModelNet40Dataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            task=task,
        )
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = modelnet40_collate

    elif config["dataset"] == "S3DIS":

        test_dataset = S3DISDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            task=task,
        )
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = s3dis_collate

    elif config["dataset"] == "Toronto3D":

        test_dataset = Toronto3DDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            task=task,
        )
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = toronto_3d_collate

    elif config["dataset"] == "SemanticKitti":

        test_dataset = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            infered_file=filename,
            task=task,
            balance_classes=False,
        )
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = semantickitti_collate

    else:
        raise ValueError("Unsupported dataset : " + config["dataset"])

    # Data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config["input"]["threads"],
        pin_memory=True,
    )

    # Calibrate samplers, one for each dataset
    test_sampler.calibration(config, test_loader, verbose=True)

    print("\nModel Preparation")
    print("*****************")

    # Define network model
    t1 = time.time()
    if config["input"]["task"] == "classification":
        net = KPCNN(config)
    elif config["input"]["task"] in ["cloud_segmentation", "slam_segmentation"]:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError("Unsupported task for testing: " + config["input"]["task"])

    # Define a visualizer class
    tester = ModelTester(net, config, chkp_path=chosen_chkp, test_path=output_path)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart test")
    print("**********\n")

    # Testing
    if config["input"]["task"] == "classification":
        tester.classification_test(net, test_loader)
    elif config["input"]["task"] == "cloud_segmentation":
        tester.cloud_segmentation_test(net, test_loader)
    elif config["input"]["task"] == "slam_segmentation":
        tester.slam_segmentation_test(net, test_loader)
    else:
        raise ValueError("Unsupported task for testing: " + config["input"]["task"])
