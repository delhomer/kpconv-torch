from pathlib import Path
import os
import time

from kpconv_torch.utils.config import load_config
import numpy as np
from torch.utils.data import DataLoader

from kpconv_torch.datasets.ModelNet40 import (
    ModelNet40Collate,
    ModelNet40Dataset,
    ModelNet40Sampler,
)
from kpconv_torch.datasets.NPM3D import (
    NPM3DCollate,
    NPM3DDataset,
    NPM3DSampler,
)
from kpconv_torch.datasets.S3DIS import (
    S3DISCollate,
    S3DISDataset,
    S3DISSampler,
)
from kpconv_torch.datasets.SemanticKitti import (
    SemanticKittiCollate,
    SemanticKittiDataset,
    SemanticKittiSampler,
)
from kpconv_torch.datasets.Toronto3D import (
    Toronto3DCollate,
    Toronto3DDataset,
    Toronto3DSampler,
)
from kpconv_torch.models.architectures import KPCNN, KPFCNN
from kpconv_torch.utils.trainer import get_train_save_path, ModelTrainer


def main(args):
    train(
        args.datapath,
        args.chosen_log,
        args.output_dir,
        args.dataset,
        args.max_epoch,
        args.checkpoint_gap,
        args.epoch_steps,
    )


def train(
    datapath: Path,
    chosen_log: Path,
    output_dir: Path,
    dataset: str,
    max_epoch: int = None,
    checkpoint_gap: int = None,
    epoch_steps: int = None,
) -> None:

    ############################
    # Initialize the environment
    ############################
    start = time.time()
    # Set which gpu is going to be used
    GPU_ID = "0"

    # Set GPU visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    ##############
    # Prepare Data
    ##############
    print()
    print("Data Preparation")
    print("****************")

    train_save_path = get_train_save_path(output_dir, chosen_log)

    # Test if the provided dataset (passed to the -d option)
    # corresponds to the one of the trained model (passed to the -l option)
    if chosen_log:
        config = load_config(Path(train_save_path / "config.yml"), dataset)
        if config["model"]["dataset"] != dataset:
            t1 = config["model"]["dataset"]
            raise ValueError(
                f"Trained model dataset ({t1}) " f"does not match provided dataset ({dataset})."
            )
        chosen_log = None

    # Initialize datasets and samplers
    if dataset == "ModelNet40":
        train_dataset = ModelNet40Dataset(
            config_file_path="config.yml", datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = ModelNet40Dataset(
            config_file_path="config.yml",
            datapath=datapath,
            chosen_log=chosen_log,
            task="validate",
        )
        train_sampler = ModelNet40Sampler(train_dataset, balance_labels=True)
        test_sampler = ModelNet40Sampler(test_dataset, balance_labels=True)
        collate_fn = ModelNet40Collate
    elif dataset == "NPM3D":
        train_dataset = NPM3DDataset(
            config_file_path="config.yml", datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = NPM3DDataset(
            config_file_path="config.yml",
            datapath=datapath,
            chosen_log=chosen_log,
            task="validate",
        )
        train_sampler = NPM3DSampler(train_dataset)
        test_sampler = NPM3DSampler(test_dataset)
        collate_fn = NPM3DCollate
    elif dataset == "S3DIS":
        train_dataset = S3DISDataset(
            config_file_path="config.yml", datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = S3DISDataset(
            config_file_path="config.yml",
            datapath=datapath,
            chosen_log=chosen_log,
            task="validate",
        )
        train_sampler = S3DISSampler(train_dataset)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif dataset == "SemanticKitti":
        train_dataset = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            task="train",
            balance_classes=True,
        )
        test_dataset = SemanticKittiDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            task="validate",
            balance_classes=False,
        )
        train_sampler = SemanticKittiSampler(train_dataset)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    elif dataset == "Toronto3D":
        train_dataset = Toronto3DDataset(
            config=config, datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = Toronto3DDataset(
            config=config, datapath=datapath, chosen_log=chosen_log, task="validate"
        )
        train_sampler = Toronto3DSampler(train_dataset)
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = Toronto3DCollate
    else:
        raise ValueError("Unsupported dataset : " + dataset)

    # Initialize the dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config["model"]["input_threads"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config["model"]["input_threads"],
        pin_memory=True,
    )

    if dataset == "SemanticKitti":
        # Calibrate max_in_point value
        train_sampler.calib_max_in(config, train_loader, verbose=True)
        test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    train_sampler.calibration(train_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    print("\nModel Preparation")
    print("*****************")

    # Define network model
    t1 = time.time()
    if dataset == "ModelNet40":
        net = KPCNN(config)
    else:
        net = KPFCNN(config, train_dataset.label_values, train_dataset.ignored_labels)

    debug = False
    if debug:
        print("\n*************************************\n")
        print(net)
        print("\n*************************************\n")
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print("\n*************************************\n")
        print(
            "Model size %i"
            % sum(param.numel() for param in net.parameters() if param.requires_grad)
        )
        print("\n*************************************\n")

    # Choose index of checkpoint to start from. If None, uses the latest chkp.
    chkp_idx = None
    if chosen_log is not None:
        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join(chosen_log, "checkpoints")
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == "chkp"]

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = "current_chkp.tar"
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(chkp_path, chosen_chkp)

        # Define a trainer class
        trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, train_save_path=train_save_path)
        print(f"Done in {time.time() - t1:.1f}s\n")

        print("\nStart training")
        print("**************")

        # Training
        trainer.train(net, train_loader, test_loader, chosen_log, config)
    else:
        chosen_chkp = None

    end = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(end - start)))
