"""
Dataset preprocess functions

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915

from pathlib import Path

from kpconv_torch.datasets.modelnet40_dataset import ModelNet40Dataset
from kpconv_torch.datasets.npm3d_dataset import NPM3DDataset
from kpconv_torch.datasets.s3dis_dataset import S3DISDataset
from kpconv_torch.datasets.semantickitti_dataset import SemanticKittiDataset
from kpconv_torch.datasets.toronto3d_dataset import Toronto3DDataset

from kpconv_torch.utils.config import load_config


def main(args):
    """
    Launch the testing from the CLI arguments
    """
    preprocess(args.datapath, args.configfile)


def preprocess(datapath: Path, configfile_path: Path) -> None:
    """
    Dataset preprocessing
    :param datapath: path to the data folder
    :param configfile: path to the config file
    """
    # Option: set which gpu is going to be used and set the GPU visible device
    # By modifying the cuda_visible_devices environment variable

    # Prepare Data
    print()
    print("Data Preparation")
    print("****************")

    # Test if the provided dataset (passed to the -d option)
    # corresponds to the one of the config file to use
    config = load_config(configfile_path)

    # Initialize datasets
    if config["dataset"] == "ModelNet40":
        # Training
        _ = ModelNet40Dataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = ModelNet40Dataset(config=config, datapath=datapath, task="validate")
    elif config["dataset"] == "NPM3D":
        # Training
        _ = NPM3DDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = NPM3DDataset(config=config, datapath=datapath, task="validate")
    elif config["dataset"] == "S3DIS":
        # Training
        _ = S3DISDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = S3DISDataset(config=config, datapath=datapath, task="validate")
    elif config["dataset"] == "SemanticKitti":
        # Training
        _ = SemanticKittiDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            task="validate",
            balance_classes=False,
        )
    elif config["dataset"] == "Toronto3D":
        # Training
        _ = Toronto3DDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = Toronto3DDataset(config=config, datapath=datapath, task="validate")
