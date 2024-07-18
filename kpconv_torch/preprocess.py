from pathlib import Path

from kpconv_torch.datasets.ModelNet40 import (
    ModelNet40Dataset,
)
from kpconv_torch.datasets.NPM3D import (
    NPM3DDataset,
)
from kpconv_torch.datasets.S3DIS import (
    S3DISDataset,
)
from kpconv_torch.datasets.SemanticKitti import (
    SemanticKittiDataset,
)
from kpconv_torch.datasets.Toronto3D import (
    Toronto3DDataset,
)

from kpconv_torch.utils.config import load_config


def main(args):
    preprocess(args.datapath, args.dataset)


def preprocess(datapath: Path, dataset: str) -> None:
    # Option: set which gpu is going to be used and set the GPU visible device
    # By modifying the CUDA_VISIBLE_DEVICES environment variable

    ##############
    # Prepare Data
    ##############
    print()
    print("Data Preparation")
    print("****************")

    config_file_path = "config.yml"

    # Test if the provided dataset (passed to the -d option)
    # corresponds to the one of the config file to use
    config = load_config(config_file_path, dataset)
    if config["dataset"] != dataset:
        t1 = config["model"]["dataset"]
        raise ValueError(
            f"Trained model dataset ({t1}) " f"does not match provided dataset ({dataset})."
        )

    # Initialize datasets
    if dataset == "ModelNet40":
        # Training
        _ = ModelNet40Dataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = ModelNet40Dataset(config=config, datapath=datapath, task="validate")
    elif dataset == "NPM3D":
        # Training
        _ = NPM3DDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = NPM3DDataset(config=config, datapath=datapath, task="validate")
    elif dataset == "S3DIS":
        # Training
        _ = S3DISDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = S3DISDataset(config=config, datapath=datapath, task="validate")
    elif dataset == "SemanticKitti":
        # Training
        _ = SemanticKittiDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            task="validate",
            balance_classes=False,
        )
    elif dataset == "Toronto3D":
        # Training
        _ = Toronto3DDataset(config=config, datapath=datapath, task="train")
        # Validation
        _ = Toronto3DDataset(config=config, datapath=datapath, task="validate")
