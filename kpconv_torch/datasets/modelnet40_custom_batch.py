"""
ModelNet40CustomBatch class

@author: Hugues THOMAS, Oslandia
@date: july 2024

"""

# pylint: disable=R0913, R0914, R0912, R0902, R0915, E0401, C0103

import torch


class ModelNet40CustomBatch:
    """
    Custom batch definition with memory pinning for ModelNet40
    """

    def __init__(self, input_list):
        """
        :param input_list
        """

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        layers = (len(input_list) - 5) // 4

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + layers]]
        ind += layers
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + layers]]
        ind += layers
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + layers]]
        ind += layers
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind : ind + layers]]
        ind += layers
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.model_inds = torch.from_numpy(input_list[ind])

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()

        return self

    def to(self, device):
        """
        :param device
        """

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)

    def unstack_points(self, layer=None):
        """
        Unstack the points
        """
        return self.unstack_elements("points", layer)

    def unstack_neighbors(self, layer=None):
        """
        Unstack the neighbors indices
        """
        return self.unstack_elements("neighbors", layer)

    def unstack_pools(self, layer=None):
        """
        Unstack the pooling indices
        """
        return self.unstack_elements("pools", layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer.

        If no layer is given, then return all layers.
        """

        if element_name == "points":
            elements = self.points
        elif element_name == "neighbors":
            elements = self.neighbors
        elif element_name == "pools":
            elements = self.pools[:-1]
        else:
            raise ValueError(f"Unknown element name: {element_name}")

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i_0 = 0
                p_list = []
                if element_name == "pools":
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i_0 : i_0 + length]
                    if element_name == "neighbors":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i_0
                    elif element_name == "pools":
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i_0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def modelnet40_collate(batch_data):
    """
    docstring to do
    """
    return ModelNet40CustomBatch(batch_data)
