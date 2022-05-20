import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import FAUST

import experiments.exp_utils
from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from gem_cnn.nn.pool import ParallelTransportPool
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.matrix_features_transform import MatrixFeaturesTransform
from gem_cnn.transform.multiscale_radius_graph import MultiscaleRadiusGraph
from gem_cnn.transform.scale_mask import ScaleMask
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh


def construct_loaders(args):
    # --------------- Dataset transformations ---------------
    # Define transformations to be performed on the dataset
    max_order = args.max_order

    # Number of rings in the radial profile
    n_rings = args.n_rings

    # Ratios used for pooling
    ratios = [1, 0.25, 0.25]

    # Number of meshes per batch
    batch_size = args.batch_size

    radii = [0.05, 0.14, 0.28]

    # Transformation that computes a multi-scale radius graph and precomputes the logarithmic map.
    pre_transform = T.Compose(
        (
            compute_normals_edges_from_mesh,
            MultiscaleRadiusGraph(ratios, radii, max_neighbours=32),
            SimpleGeometry(),
            MatrixFeaturesTransform(),
        )
    )

    scale_transforms = [
        T.Compose((ScaleMask(i), GemPrecomp(n_rings, max_order, max_r=radii[i])))
        for i in range(3)
    ]

    # --------------- Set dataset path ---------------
    # Provide a path to load and store the dataset.
    path = exp_utils.get_dataset_path("faust_pool")
    # Monkey patch to change processed dir, to allow for direct and pool pre-processing
    FAUST.processed_dir = osp.join(path, "processed_pool")  # noqa

    # --------------- Load datasets ---------------
    if args.verbose:
        print("Creating datasets")
    train_dataset = FAUST(path, train=True, pre_transform=pre_transform)
    test_dataset = FAUST(path, train=False, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    num_nodes = train_dataset[0].num_nodes

    return scale_transforms, train_loader, test_loader, num_nodes


def main(args):

    exp_utils.change_random_seed(args.seed)

    # --------------- Dataset ---------------
    scale_transforms, train_loader, test_loader, num_nodes = construct_loaders(args)

    # --------------- Model structure ---------------
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Pre-transform, linear layer

            width = 16

            kwargs = dict(
                n_rings=args.n_rings,
                band_limit=args.max_order,
                num_samples=7,
                checkpoint=True,
                batch=100_000,
            )

            # Pool
            self.pool1 = ParallelTransportPool(1, unpool=False)
            self.pool2 = ParallelTransportPool(2, unpool=False)
            self.unpool1 = ParallelTransportPool(1, unpool=True)
            self.unpool2 = ParallelTransportPool(2, unpool=True)

            # Stack 1, level 0
            self.conv11 = GemResNetBlock(7, width, 2, args.max_order, **kwargs)
            self.conv12 = GemResNetBlock(
                width, width, args.max_order, args.max_order, **kwargs
            )

            # Stack 2, level 1
            self.conv21 = GemResNetBlock(
                width, width, args.max_order, args.max_order, **kwargs
            )
            self.conv22 = GemResNetBlock(
                width, width, args.max_order, args.max_order, **kwargs
            )

            # Stack 3, level 2
            self.conv31 = GemResNetBlock(
                width, width, args.max_order, args.max_order, **kwargs
            )
            self.conv32 = GemResNetBlock(
                width, width, args.max_order, args.max_order, **kwargs
            )

            # Stack 4, level 1
            self.conv41 = GemResNetBlock(
                2 * width, width, args.max_order, args.max_order, **kwargs
            )
            self.conv42 = GemResNetBlock(
                width, width, args.max_order, args.max_order, **kwargs
            )

            # Stack 5, level 0
            self.conv51 = GemResNetBlock(
                2 * width, width, args.max_order, args.max_order, **kwargs
            )
            self.conv52 = GemResNetBlock(width, width, args.max_order, 0, **kwargs)

            # Dense final layers
            self.lin1 = nn.Linear(width, 256)
            self.lin2 = nn.Linear(256, num_nodes)

        def forward(self, data):
            data0 = scale_transforms[0](data)
            data1 = scale_transforms[1](data)
            data2 = scale_transforms[2](data)
            attr0 = (data0.edge_index, data0.precomp, data0.connection)
            attr1 = (data1.edge_index, data1.precomp, data1.connection)
            attr2 = (data2.edge_index, data2.precomp, data2.connection)

            x = data.matrix_features

            # Stack 1
            # Select only the edges and precomputed components of the first scale
            x = self.conv11(x, *attr0)
            x = x_l0 = self.conv12(x, *attr0)

            x = self.pool1(x, data)

            x = self.conv21(x, *attr1)
            x = x_l1 = self.conv22(x, *attr1)

            x = self.pool2(x, data)

            x = self.conv31(x, *attr2)
            x = self.conv32(x, *attr2)

            x = self.unpool2(x, data)
            x = torch.cat((x, x_l1), dim=1)

            x = self.conv41(x, *attr1)
            x = self.conv42(x, *attr1)

            x = self.unpool1(x, data)
            x = torch.cat((x, x_l0), dim=1)

            x = self.conv51(x, *attr0)
            x = self.conv52(x, *attr0)

            # Take trivial feature
            x = x[:, :, 0]

            x = F.relu(self.lin1(x))
            x = F.dropout(x, training=self.training)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

    if args.verbose:
        print("Creating model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    # --------------- Train ---------------
    target = (
        torch.arange(num_nodes, dtype=torch.long, device=device)
        .expand(args.batch_size, num_nodes)
        .flatten()
    )
    prepare_batch_fn = exp_utils.core_prepare_batch(target)

    engine = exp_utils.GEMEngine(model, optimizer, criterion, device, prepare_batch_fn)
    engine.set_epoch_loggers(val_loader=test_loader)

    if args.use_wandb:
        wandb_logger = engine.create_wandb_logger(
            task_name="faust", tags=["pool"], config=vars(args)
        )

    engine.trainer.run(train_loader, max_epochs=args.epochs)


def parse_arguments():
    parser = argparse.ArgumentParser(description="FAUST Parser")

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--n_rings", default=2, type=int)
    parser.add_argument("--max_order", default=2, type=int)

    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate")
    parser.add_argument(
        "-bs", "--batch_size", default=4, type=int, help="Number of meshes per batch"
    )

    parser.add_argument("-wandb", dest="use_wandb", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":

    try_args = parse_arguments()
    main(try_args)
