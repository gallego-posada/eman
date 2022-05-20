# This code was modified by The EMAN Authors to fix a bug in the original
# GEM-CNN implementation which made the GEM-CNN model non-equivariant.

# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
"""
Main convolution.

Data is arranged as:

x[number of vertices in batch, number of channels, dimensionality of representation]
"""
from functools import partial

import torch
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot

from gem_cnn.utils.einsum import einsum
from gem_cnn.utils.kernel import build_kernel
from gem_cnn.utils.rep_act import rep_act


class GemConv(MessagePassing):
    """
    GEM Convolution

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        in_order (int): order of input
        out_order (int): order of output
        n_rings (int): number of radial rings
        band_limit (int, optional): maximum theta frequency used
        batch (int, optional): compute edges in batches, checkpointed to save memory
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_order,
        out_order,
        n_rings,
        band_limit=None,
        batch=None,
        equiv_bias=False,
    ):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_order = in_order
        self.out_order = out_order
        self.n_rings = n_rings
        # self.kernel has shape [n_bases, 2 * band_limit + 1, 2 * order_out + 1, 2 * order_in + 1]
        self.register_buffer(
            "kernel",
            torch.tensor(
                build_kernel(in_order, out_order, band_limit), dtype=torch.float32
            ),
        )
        self.weight = Parameter(
            torch.Tensor(self.kernel.shape[0], n_rings, out_channels, in_channels)
        )
        self.bias = Parameter(torch.Tensor(out_channels))
        self.batch = batch

        self.equiv_bias = equiv_bias
        if self.equiv_bias:
            self.ang_bias1 = Parameter(torch.Tensor(out_channels))
            self.ang_bias2 = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        torch.nn.init.normal_(self.bias)
        if self.equiv_bias:
            torch.nn.init.normal_(self.ang_bias1, 0, 0.1)
            torch.nn.init.normal_(self.ang_bias2, 0, 0.1)

    @staticmethod
    def get_rot_matrix(bias):
        cos_, sin_ = torch.cos(bias), torch.sin(bias)
        return torch.stack([cos_, -sin_, sin_, cos_]).permute(1, 0).reshape(-1, 2, 2)

    def forward(self, x, edge_index, precomp, connection):
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == 2 * self.in_order + 1
        assert precomp.dim() == 3
        out = self.propagate(
            edge_index=edge_index, x=x, precomp=precomp, connection=connection
        )

        return out

    def message(self, x_j, precomp, connection):
        """
        :param x_j: [n_edges, in_channels, 2*in_order+1]
        :param precomp [n_edges, 2*band_limit+1, n_rings]
        :param connection: [n_edges]
        :return: [num_v, out_channels, 2*out_order+1]
        """
        assert (
            self.kernel.shape[1] <= precomp.shape[1]
        ), "Kernel set with higher band-limit than precompute"
        precomp = precomp[:, : self.kernel.shape[1]]

        # x_j = x[edge_index[1, :]]
        # parallel transport neighbors (x_j) using angles (connection)
        x_j_transported = rep_act(x_j, connection)
        if self.batch is None:
            # implementation of neighbor kernel, where precomp contains sines and cosines, kernel is a binary tensor
            # choosing appropriate sine or cosine values and weight is the learnable parameter
            y = einsum(
                "ejm,efr,bfnm,brij->ein",
                x_j_transported,
                precomp,
                self.kernel,
                self.weight,
            )
        else:
            ys = []
            for i in range(0, x_j.shape[0], self.batch):
                # implementation of neighbor kernel, where precomp contains sines and cosines, kernel is a binary tensor
                # choosing appropriate sine or cosine values and weight is the learnable parameter
                y = checkpoint(
                    partial(einsum, "ejm,efr,bfnm,brij->ein"),
                    x_j_transported[i : i + self.batch],
                    precomp[i : i + self.batch],
                    self.kernel,
                    self.weight,
                )
                ys.append(y)
            y = torch.cat(ys)

        if self.equiv_bias:

            y[:, :, 0] += self.bias[None, :]
            if y.shape[2] > 1:

                # Applying rotational/angular bias for rho_1, rho_2 etc features
                # to preserve equivariance
                rot_matrix1 = self.get_rot_matrix(self.ang_bias1)
                rot_matrix2 = self.get_rot_matrix(self.ang_bias2)

                # Note that tensor.clone is a autodiff compatible operation
                y[:, :, 1:3] = torch.einsum(
                    "cij, ncj -> nci", rot_matrix1, y[:, :, 1:3].clone()
                )
                y[:, :, 3:5] = torch.einsum(
                    "cij, ncj -> nci", rot_matrix2, y[:, :, 3:5].clone()
                )
        else:
            y += self.bias[None, :, None]

        return y
