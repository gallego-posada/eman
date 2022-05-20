import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_isolated_nodes

from eman.nn.eman_res_net_block import EmanAttResNetBlock
from eman.transform.rel_tan_features import RelTanTransform
from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from gem_cnn.transform.gem_precomp import GemPrecomp
from spiralnet.spiralconv import SpiralConv


class GemCNN(torch.nn.Module):
    def __init__(self, args):
        super(GemCNN, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        self.transform = GemPrecomp(args.n_rings, args.max_order)
        self.conv1 = GemResNetBlock(3, width, 0, args.max_order, **kwargs)
        self.conv2 = GemResNetBlock(
            width, width, args.max_order, args.max_order, **kwargs
        )
        self.conv3 = GemResNetBlock(width, width, args.max_order, 0, **kwargs)

        # Dense final layers
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        data0 = self.transform(data)
        attr0 = (data0.edge_index, data0.precomp, data0.connection)

        # takes positions as input features
        x = data.pos[:, :, None]

        x = self.conv1(x, *attr0)
        x = self.conv2(x, *attr0)
        x = self.conv3(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = torch.mean(x, dim=0, keepdim=True)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class RelTanGemCNN(torch.nn.Module):
    def __init__(self, args):
        super(RelTanGemCNN, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        self.transform = GemPrecomp(args.n_rings, args.max_order)
        self.rel_transform = RelTanTransform(args)
        input_channels = len(args.rel_power_list)
        self.conv1 = GemResNetBlock(input_channels, width, 1, args.max_order, **kwargs)
        self.conv2 = GemResNetBlock(
            width, width, args.max_order, args.max_order, **kwargs
        )
        self.conv3 = GemResNetBlock(width, width, args.max_order, 0, **kwargs)

        # Dense final layers
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.null_isolated = hasattr(args, "null_isolated") and args.null_isolated

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        data0 = self.transform(data)
        # rel_transform adds rel_tang_feat (check Sec. 4 in the draft) feature to data
        self.rel_transform(data)
        attr0 = (data0.edge_index, data0.precomp, data0.connection)

        # x = data.pos[:, :, None]
        # takes relative tangent features as input features
        x = data.rel_tang_feat

        # Setting the features of isolated nodes to 0
        if self.null_isolated:
            non_isol_mask = remove_isolated_nodes(data0.edge_index)[-1]
            x[~non_isol_mask] = 0.0

        x = self.conv1(x, *attr0)
        x = self.conv2(x, *attr0)
        x = self.conv3(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = torch.mean(x, dim=0, keepdim=True)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class EMAN(torch.nn.Module):
    def __init__(self, args):
        super(EMAN, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        self.transform = GemPrecomp(args.n_rings, args.max_order)
        self.conv1 = EmanAttResNetBlock(3, width, 0, args.max_order, **kwargs)
        self.conv2 = EmanAttResNetBlock(
            width, width, args.max_order, args.max_order, **kwargs
        )
        self.conv3 = EmanAttResNetBlock(width, width, args.max_order, 0, **kwargs)

        # Dense final layers
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        data0 = self.transform(data)
        attr0 = (data0.edge_index, data0.precomp, data0.precomp_self, data0.connection)

        # takes positions as input features
        x = data.pos[:, :, None]

        x = self.conv1(x, *attr0)
        x = self.conv2(x, *attr0)
        x = self.conv3(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = torch.mean(x, dim=0, keepdim=True)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class RelTanEMAN(torch.nn.Module):
    def __init__(self, args):
        super(RelTanEMAN, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        self.transform = GemPrecomp(args.n_rings, args.max_order)
        self.rel_transform = RelTanTransform(args)
        input_channels = len(args.rel_power_list)
        self.conv1 = EmanAttResNetBlock(
            input_channels, width, 1, args.max_order, **kwargs
        )
        self.conv2 = EmanAttResNetBlock(
            width, width, args.max_order, args.max_order, **kwargs
        )
        self.conv3 = EmanAttResNetBlock(width, width, args.max_order, 0, **kwargs)

        # Dense final layers
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.null_isolated = hasattr(args, "null_isolated") and args.null_isolated

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        data0 = self.transform(data)
        # rel_transform adds rel_tang_feat (check Sec. 4 in the draft) feature to data
        self.rel_transform(data)
        attr0 = (data0.edge_index, data0.precomp, data0.precomp_self, data0.connection)

        # x = data.pos[:, :, None]
        # takes relative tangent features as input features
        x = data.rel_tang_feat

        # Setting the features of isolated nodes to 0
        if self.null_isolated:
            non_isol_mask = remove_isolated_nodes(data0.edge_index)[-1]
            x[~non_isol_mask] = 0.0

        x = self.conv1(x, *attr0)
        x = self.conv2(x, *attr0)
        x = self.conv3(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = torch.mean(x, dim=0, keepdim=True)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class SpiralNet(torch.nn.Module):
    def __init__(self, args, indices):
        super(SpiralNet, self).__init__()
        in_channels = args.spiral_net_input_channels
        num_classes = args.spiral_net_num_classes
        self.fc0 = nn.Linear(in_channels, 16)
        self.conv1 = SpiralConv(16, 32, indices)
        self.conv2 = SpiralConv(32, 64, indices)
        self.conv3 = SpiralConv(64, 128, indices)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, data):
        x = data.pos
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
