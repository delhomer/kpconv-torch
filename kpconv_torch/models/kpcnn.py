"""
Class KPCNN

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, E0401, C0103

from kpconv_torch.models.architecture_functions import p2p_fitting_regularizer
from kpconv_torch.models.blocks import block_decider, nn, torch, UnaryBlock


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super().__init__()

        # Network opperations
        # Current radius of convolution and feature dimension
        layer = 0
        r = config["kpconv"]["first_subsampling_dl"] * config["kpconv"]["conv_radius"]
        in_dim = config["input"]["features_dim"]
        out_dim = config["model"]["first_features_dim"]
        self.K = config["kpconv"]["num_kernel_points"]
        self.num_classes = len(config["model"]["label_to_names"])

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        for block in config["model"]["architecture"]:

            # Check equivariance
            if ("equivariant" in block) and (out_dim % 3 != 0):
                raise ValueError("Equivariant block but features dimension is not a factor of 3")

            # Detect upsampling block to stop
            if "upsample" in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block, r, in_dim, out_dim, layer, config))

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

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, self.num_classes, False, 0, no_relu=True)

        # Network Losses
        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config["kpconv"]["deform_fitting_mode"]
        self.deform_fitting_power = config["kpconv"]["deform_fitting_power"]
        self.deform_lr_factor = config["train"]["deform_lr_factor"]
        self.repulse_extent = config["train"]["repulse_extent"]
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
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

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == "point2point":
            self.reg_loss = p2p_fitting_regularizer(self)
            # Combined loss
            return self.output_loss + self.reg_loss

        if self.deform_fitting_mode == "point2plane":
            raise ValueError("point2plane fitting mode not implemented yet.")

        raise ValueError("Unknown fitting mode: " + self.deform_fitting_mode)

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total
