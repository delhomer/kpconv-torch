"""
Models training functions

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915

from pathlib import Path
import os
import time

import numpy as np
from torch.utils.data import DataLoader

from kpconv_torch.utils.config import load_config
from kpconv_torch.datasets.modelnet40_dataset import ModelNet40Dataset
from kpconv_torch.datasets.modelnet40_sampler import ModelNet40Sampler
from kpconv_torch.datasets.modelnet40_custom_batch import modelnet40_collate
from kpconv_torch.datasets.npm3d_dataset import NPM3DDataset
from kpconv_torch.datasets.npm3d_sampler import NPM3DSampler
from kpconv_torch.datasets.npm3d_custom_batch import npm3d_collate
from kpconv_torch.datasets.semantickitti_dataset import SemanticKittiDataset
from kpconv_torch.datasets.semantickitti_sampler import SemanticKittiSampler
from kpconv_torch.datasets.semantickitti_custom_batch import semantickitti_collate
from kpconv_torch.datasets.s3dis_dataset import S3DISDataset
from kpconv_torch.datasets.s3dis_sampler import S3DISSampler
from kpconv_torch.datasets.s3dis_custom_batch import s3dis_collate
from kpconv_torch.datasets.toronto3d_dataset import Toronto3DDataset
from kpconv_torch.datasets.toronto3d_sampler import Toronto3DSampler
from kpconv_torch.datasets.toronto3d_custom_batch import toronto3d_collate
from kpconv_torch.models.architectures import KPCNN, KPFCNN
from kpconv_torch.utils.trainer import get_train_save_path, ModelTrainer


def main(args):
    """
    Launch the training from the CLI arguments
    """
    train(
        args.datapath,
        args.configfile,
        args.chosen_log,
        args.output_dir,
    )


def train(
    datapath: Path,
    configfile: Path,
    chosen_log: Path,
    output_dir: Path,
) -> None:
    """
    Model training
    :param datapath: path to the data folder
    :param configfile: path to the config file
    :param chosen_log: path to an already trained model, for the training to be continued
    :param output_dir: path to the folder which will contain the results of the training
    """
    # Initialize the environment
    start = time.time()
    # Set which gpu is going to be used
    gpu_id = "0"

    # Set GPU visible device
    os.environ["cuda_visible_devices"] = gpu_id

    # Prepare Data
    print()
    print("Data Preparation")
    print("****************")

    train_save_path = get_train_save_path(output_dir, chosen_log)
    if chosen_log is None:
        config_file_path = configfile
    else:
        config_file_path = Path(train_save_path / "config.yml")
    config = load_config(config_file_path)

    # Initialize datasets and samplers
    if config["dataset"] == "ModelNet40":
        train_dataset = ModelNet40Dataset(
            config=config, datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = ModelNet40Dataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            task="validate",
        )
        train_sampler = ModelNet40Sampler(train_dataset, balance_labels=True)
        test_sampler = ModelNet40Sampler(test_dataset, balance_labels=True)
        collate_fn = modelnet40_collate
    elif config["dataset"] == "NPM3D":
        train_dataset = NPM3DDataset(
            config=config, datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = NPM3DDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            task="validate",
        )
        train_sampler = NPM3DSampler(train_dataset)
        test_sampler = NPM3DSampler(test_dataset)
        collate_fn = npm3d_collate
    elif config["dataset"] == "S3DIS":
        train_dataset = S3DISDataset(
            config=config, datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = S3DISDataset(
            config=config,
            datapath=datapath,
            chosen_log=chosen_log,
            task="validate",
        )
        train_sampler = S3DISSampler(train_dataset)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = s3dis_collate
    elif config["dataset"] == "SemanticKitti":
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
        collate_fn = semantickitti_collate
    elif config["dataset"] == "Toronto3D":
        train_dataset = Toronto3DDataset(
            config=config, datapath=datapath, chosen_log=chosen_log, task="train"
        )
        test_dataset = Toronto3DDataset(
            config=config, datapath=datapath, chosen_log=chosen_log, task="validate"
        )
        train_sampler = Toronto3DSampler(train_dataset)
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = toronto3d_collate
    else:
        raise ValueError("Unsupported dataset : " + config["dataset"])

    # Initialize the dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config["input"]["threads"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config["input"]["threads"],
        pin_memory=True,
    )

    if config["dataset"] == "SemanticKitti":
        # Calibrate max_in_point value
        train_sampler.calib_max_in(config, train_loader, verbose=True)
        test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    train_sampler.calibration(config, train_loader, verbose=True)
    test_sampler.calibration(config, test_loader, verbose=True)

    print("\nModel Preparation")
    print("*****************")

    # Define network model
    t1 = time.time()
    if config["dataset"] == "ModelNet40":
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
            f"Model size \
              {sum(param.numel() for param in net.parameters() if param.requires_grad)}"
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
    else:
        chosen_chkp = None

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, train_save_path=train_save_path)
    print(f"Done in {time.time() - t1:.1f}s\n")

    print("\nStart training")
    print("**************")

    # Training
    trainer.train(net, train_loader, test_loader, chosen_log)

    end = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(end - start)))
