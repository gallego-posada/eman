from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from eman.utils.kernel import build_self_kernel
from gem_cnn.utils.einsum import einsum
from gem_cnn.utils.kernel import build_kernel
from gem_cnn.utils.rep_act import rep_act


class EmanAttLayer(nn.Module):
    """
    EMAN Attention Layer: computes attention for EMANAttConv

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        in_order (int): order of input
        out_order (int): order of output
        n_rings (int): number of radial rings
        band_limit (int, optional): maximum theta frequency used
        batch (int, optional): compute edges in batches, checkpointed to save memory
        n_heads (int, optional): number of heads in attention module
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
        n_heads=2,
        equiv_bias=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_order = in_order
        self.out_order = out_order
        self.n_rings = n_rings
        self.n_heads = n_heads
        self.d_k = self.out_channels * (2 * self.out_order + 1) // self.n_heads
        self.rho_in = in_channels * [i for i in range(in_order + 1)]
        self.rho_out = out_channels * [i for i in range(out_order + 1)]
        self.register_buffer(
            "self_kernel",
            torch.tensor(
                build_self_kernel(in_order, out_order, band_limit), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "neigh_kernel",
            torch.tensor(
                build_kernel(in_order, out_order, band_limit), dtype=torch.float32
            ),
        )
        self.self_weight = Parameter(
            torch.Tensor(self.self_kernel.shape[0], n_rings, out_channels, in_channels)
        )
        self.neigh_weight = Parameter(
            torch.Tensor(self.neigh_kernel.shape[0], n_rings, out_channels, in_channels)
        )
        # self.self_lin = nn.Linear(in_channels * (2 * in_order + 1), out_channels * (2 * out_order + 1))
        self.self_bias = Parameter(torch.Tensor(out_channels))
        self.neigh_bias = Parameter(torch.Tensor(out_channels))
        self.batch = batch
        self.equiv_bias = equiv_bias

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.self_weight)
        glorot(self.neigh_weight)
        zeros(self.self_bias)
        zeros(self.neigh_bias)

    def forward(self, x, edge_index, precomp, precomp_self, connection):
        assert (
            self.neigh_kernel.shape[1] <= precomp.shape[1]
        ), "Neighbor Kernel set with higher band-limit than precompute"
        precomp = precomp[:, : self.neigh_kernel.shape[1]]

        # assert (
        #         self.self_kernel.shape[1] <= precomp.shape[1]
        # ), "Self Kernel set with higher band-limit than precompute"
        # precomp_self = precomp_self[:, : self.self_kernel.shape[1]]

        # x_i is the features at source side of the edges
        x_i = x[edge_index[0, :]]
        # x_j is the features at destination side of the edges
        x_j = x[edge_index[1, :]]
        x_j_transported = rep_act(x_j, connection)

        if self.batch is None:
            shape = (x_j.shape[0], self.n_heads, -1)
            # implementation of neighbor kernel, where precomp contains sines and cosines, kernel is a binary tensor
            # choosing appropriate sine or cosine values and weight is the learnable parameter
            y_dst = einsum(
                "ejm,efr,bfnm,brij->ein",
                x_j_transported,
                precomp,
                self.neigh_kernel,
                self.neigh_weight,
            )
            y_dst = y_dst.reshape(shape)
            # implementation of self kernel, where precomp_self contains sines and cosines with theta=0
            # (since self kernel), self_kernel is a binary tensor choosing appropriate sine or cosine values
            # (with theta 0) and weight is the learnable parameter
            y_src = einsum(
                "ejm,efr,bfnm,brij->ein",
                x_i,
                precomp_self,
                self.self_kernel,
                self.self_weight,
            )
            y_src = y_src.reshape(shape)

            prods = torch.sum(y_src * y_dst, dim=2) / np.sqrt(
                self.d_k
            )  # dims = [n_edges, n_heads]
            # compute degree of nodes
            deg = (scatter_add(torch.ones_like(edge_index[0]), edge_index[0]))[
                edge_index[0]
            ].unsqueeze(dim=1)
            # compute attention
            attention = deg * softmax(prods, edge_index[0], dim=0)
        else:
            attentions = []
            for i in range(0, x_j.shape[0], self.batch):
                shape = (min(self.batch, x_j.shape[0] - i), self.n_heads, -1)
                # implementation of neighbor kernel, where precomp contains sines and cosines, kernel is a binary tensor
                # choosing appropriate sine or cosine values and weight is the learnable parameter
                y_dst = checkpoint(
                    partial(einsum, "ejm,efr,bfnm,brij->ein"),
                    x_j_transported[i : i + self.batch],
                    precomp[i : i + self.batch],
                    self.neigh_kernel,
                    self.neigh_weight,
                )
                y_dst = y_dst.reshape(shape)
                # implementation of self kernel, where precomp_self contains sines and cosines with theta=0
                # (since self kernel), self_kernel is a binary tensor choosing appropriate sine or cosine values
                # (with theta 0) and weight is the learnable parameter
                y_src = checkpoint(
                    partial(einsum, "ejm,efr,bfnm,brij->ein"),
                    x_i[i : i + self.batch],
                    precomp_self[i : i + self.batch],
                    self.self_kernel,
                    self.self_weight,
                )
                y_src = y_src.reshape(shape)
                prods = torch.sum(y_src * y_dst, dim=2) / np.sqrt(
                    self.d_k
                )  # dims = [n_edges, n_heads]
                # compute degree of nodes
                deg = (scatter_add(torch.ones_like(edge_index[0]), edge_index[0]))[
                    edge_index[0]
                ].unsqueeze(dim=1)
                # compute attention
                attention = deg * softmax(prods, edge_index[0], dim=0)
                attentions.append(attention)
            attention = torch.cat(attentions)
        return attention


class EmanAttConv(MessagePassing):
    """
    EMAN Convolution

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
        n_heads=1,
        equiv_bias=False,
    ):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_order = in_order
        self.out_order = out_order
        self.n_rings = n_rings
        self.n_heads = n_heads
        self.edge_index = None
        self.x = None

        self.eman_attn_layer = EmanAttLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            in_order=in_order,
            out_order=out_order,
            n_rings=n_rings,
            band_limit=band_limit,
            batch=batch,
            n_heads=n_heads,
            equiv_bias=equiv_bias,
        )
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

    def forward(self, x, edge_index, precomp, precomp_self, connection):
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == 2 * self.in_order + 1
        assert precomp.dim() == 3
        self.edge_index = edge_index
        self.x = x
        # get attention coefficients
        attention = self.eman_attn_layer(
            x, edge_index, precomp, precomp_self, connection
        )
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            precomp=precomp,
            precomp_self=precomp_self,
            connection=connection,
            attention=attention,
        )

        return out

    def message(self, x_j, precomp, precomp_self, connection, attention):
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
            shape_att = (x_j.shape[0], self.n_heads, -1)
            # implementation of neighbor kernel, where precomp contains sines and cosines, kernel is a binary tensor
            # choosing appropriate sine or cosine values and weight is the learnable parameter
            y = einsum(
                "ejm,efr,bfnm,brij->ein",
                x_j_transported,
                precomp,
                self.kernel,
                self.weight,
            )  # [N, c_out, rep_len]
            shape_y = y.shape
            y = y.reshape(shape_att)
            # multiply with attention coefficients
            attn = attention.unsqueeze(dim=2)
            y = y * attn
            y = y.reshape(shape_y)
        else:
            ys = []
            for i in range(0, x_j.shape[0], self.batch):
                shape_att = (min(self.batch, x_j.shape[0] - i), self.n_heads, -1)
                # implementation of neighbor kernel, where precomp contains sines and cosines, kernel is a binary tensor
                # choosing appropriate sine or cosine values and weight is the learnable parameter
                y = checkpoint(
                    partial(einsum, "ejm,efr,bfnm,brij->ein"),
                    x_j_transported[i : i + self.batch],
                    precomp[i : i + self.batch],
                    self.kernel,
                    self.weight,
                )  # [N, c_out, rep_len]
                shape_y = y.shape
                y = y.reshape(shape_att)
                # multiply with attention coefficients
                attention = attention.unsqueeze(dim=2)
                y = y * attention
                y = y.reshape(shape_y)
                ys.append(y)  # [bs, N, c_out, rep_len]
            y = torch.cat(ys)

        if self.equiv_bias:

            y[:, :, 0] += self.bias[None, :]
            if y.shape[2] > 1:
                # print(f"bias1: {self.ang_bias1}")
                # print(f"bias2: {self.ang_bias2}")
                # self.bias1.data, self.bias2.data = torch.clip(self.bias1, min=-np.pi, max=np.pi), torch.clip(self.bias2, min=-np.pi, max=np.pi)

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
