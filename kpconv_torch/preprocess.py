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

    # Initialize datasets
    if dataset == "ModelNet40":
        # Training
        _ = ModelNet40Dataset(config_file_path="config.yml", datapath=datapath, split="train")
        # Validation
        _ = ModelNet40Dataset(config_file_path="config.yml", datapath=datapath, split="validation")
    elif dataset == "NPM3D":
        # Training
        _ = NPM3DDataset(config_file_path="config.yml", datapath=datapath, split="train")
        # Validation
        _ = NPM3DDataset(config_file_path="config.yml", datapath=datapath, split="validation")
    elif dataset == "S3DIS":
        # Training
        _ = S3DISDataset(config_file_path="config.yml", datapath=datapath, split="train")
        # Validation
        _ = S3DISDataset(config_file_path="config.yml", datapath=datapath, split="validation")
    elif dataset == "SemanticKitti":
        # Training
        _ = SemanticKittiDataset(config_file_path="config.yml", datapath=datapath, split="train")
        # Validation
        _ = SemanticKittiDataset(
            config_file_path="config.yml",
            datapath=datapath,
            split="validation",
            balance_classes=False,
        )
    elif dataset == "Toronto3D":
        # Training
        _ = Toronto3DDataset(config_file_path="config.yml", datapath=datapath, split="train")
        # Validation
        _ = Toronto3DDataset(config_file_path="config.yml", datapath=datapath, split="validation")
