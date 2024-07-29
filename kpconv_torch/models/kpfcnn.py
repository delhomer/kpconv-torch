"""
Class KPFCNN

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, E0401, C0103

import numpy as np

from kpconv_torch.models.architecture_functions import p2p_fitting_regularizer
from kpconv_torch.models.blocks import block_decider, nn, torch, UnaryBlock


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super().__init__()

        # Parameters
        # Current radius of convolution and feature dimension
        layer = 0
        r = config["kpconv"]["first_subsampling_dl"] * config["kpconv"]["conv_radius"]
        in_dim = config["input"]["features_dim"]
        out_dim = config["model"]["first_features_dim"]
        self.K = config["kpconv"]["num_kernel_points"]
        self.C = len(lbl_values) - len(ign_lbls)

        # List Encoder blocks
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config["model"]["architecture"]):

            # Check equivariance
            if ("equivariant" in block) and (out_dim % 3 != 0):
                raise ValueError("Equivariant block but features dimension is not a factor of 3")

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ["pool", "strided", "upsample", "global"]]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if "upsample" in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            if "simple" in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if "pool" in block or "strided" in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        # List Decoder blocks
        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config["model"]["architecture"]):
            if "upsample" in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config["model"]["architecture"][start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and "upsample" in config["model"]["architecture"][start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if "upsample" in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config["model"]["first_features_dim"], False, 0)
        self.head_softmax = UnaryBlock(
            config["model"]["first_features_dim"], self.C, False, 0, no_relu=True
        )

        # Network Losses
        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config["train"]["class_w"]) > 0:
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=config["train"]["class_w"], ignore_index=-1
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config["kpconv"]["deform_fitting_mode"]
        self.deform_fitting_power = config["kpconv"]["deform_fitting_power"]
        self.deform_lr_factor = config["train"]["deform_lr_factor"]
        self.repulse_extent = config["train"]["repulse_extent"]
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

    def forward(self, batch, config):
        """
        docstring to do
        """
        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        return self.head_softmax(x, batch)

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == "point2point":
            self.reg_loss = p2p_fitting_regularizer(self)
            # Combined loss
            return self.output_loss + self.reg_loss

        if self.deform_fitting_mode == "point2plane":
            raise ValueError("point2plane fitting mode not implemented yet.")

        raise ValueError("Unknown fitting mode: " + self.deform_fitting_mode)

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = -torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
