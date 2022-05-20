import os.path as osp
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader

import experiments.exp_utils as exp_utils
import experiments.models as models
from eman.transform.permute_nodes import PermuteNodes
from experiments.exp_utils import parse_arguments
from experiments.faust_direct import FAUST
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh
from spiralnet.utils import preprocess_spiral

# sys.path.append(".")


def shuffle_backwards(shuffle):
    a = torch.arange(len(shuffle))
    a[shuffle] = torch.arange(len(shuffle))
    return a


def visualize_graph(data):
    # Converting from pytorch geometric to networkx for visualisation
    fig = plt.figure()
    if len(data.pos[0]) > 2:
        pos = data.pos[:, :-1]
    else:
        pos = data.pos
    g = torch_geometric.utils.to_networkx(data, to_undirected=False)
    nx.draw(g, pos=pos.numpy(), with_labels=True, cmap=plt.get_cmap("Set3"))

    fig.suptitle("Graph visualization")
    fig.show()
    plt.show()


def shuffle_test():
    x = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    pos = torch.tensor([[0, 0, 0], [1, 1, 0], [1, -1, 0], [2, 0, 0]], dtype=torch.float)
    face = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    face[0], face[1], face[2] = (
        x.clone()[face[0]],
        x.clone()[face[1]],
        x.clone()[face[2]],
    )
    shuffle = np.random.choice(len(pos), size=len(pos), replace=False)
    backward_shuffle = shuffle_backwards(shuffle)
    data = Data(x=x, pos=pos, face=face)
    data = compute_normals_edges_from_mesh(data)
    data.pos = data.pos[:, :-1]
    visualize_graph(data)

    x_shuffle = x.clone()[backward_shuffle]
    pos_shuffle = pos.clone()[shuffle]
    face_shuffle = face.clone()
    face_shuffle[0], face_shuffle[1], face_shuffle[2] = (
        x_shuffle.clone()[face_shuffle[0]],
        x_shuffle.clone()[face_shuffle[1]],
        x_shuffle.clone()[face_shuffle[2]],
    )
    data_shuffle = Data(x=x_shuffle, pos=pos_shuffle, face=face_shuffle)
    data_shuffle = compute_normals_edges_from_mesh(data_shuffle)
    data_shuffle.pos = data_shuffle.pos[:, :-1]
    print(f"shuffle: {shuffle}")
    print(f"backward_shuffle: {backward_shuffle}")
    visualize_graph(data_shuffle)


def test_perm_eq():
    n_rings = 2
    max_order = 4
    num_v = 6890
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle = np.random.choice(num_v, size=num_v, replace=False)
    print(f"shuffle: {shuffle}")
    backward_shuffle = shuffle_backwards(shuffle)

    gem_precomp = GemPrecomp(n_rings=n_rings, max_order=max_order)
    permute_nodes = PermuteNodes(shuffle)

    transform = T.Compose(
        [
            compute_normals_edges_from_mesh,
            SimpleGeometry(),
            gem_precomp,
        ]
    )

    transform_p = T.Compose(
        [
            permute_nodes,
            compute_normals_edges_from_mesh,
            SimpleGeometry(),
            gem_precomp,
        ]
    )

    path = exp_utils.get_dataset_path("faust_direct")
    FAUST.processed_dir = osp.join(path, "processed_direct")
    train_dataset = FAUST(path, train=True, pre_transform=transform)

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    data = next(iter(dataloader))

    data_raw = Data(pos=data.pos, face=data.face)
    data = transform(data_raw)
    data_p = transform_p(data_raw)
    assert torch.allclose(data.pos[backward_shuffle], data_p.pos, atol=1e-14)

    parser = parse_arguments()
    args = parser.parse_args()

    args.equiv_bias = False
    args.model_batch = 100_000
    args.num_nodes, args.target_dim = 6890, 6890
    args.model = "GemCNN"

    if args.model == "SpiralNet":
        args.lr = 3e-3  # SpiralNet shows poor performance with higher learning rate
        args.spiral_net_seq_length = 4
        args.spiral_net_input_channels = 3
        args.spiral_net_num_classes = num_v
        spiral_indices = preprocess_spiral(data.face.T, args.spiral_net_seq_length).to(
            device
        )
        model = models.__dict__[args.model](args, spiral_indices).to(device)
    else:
        model = models.__dict__[args.model](args).to(device)
    model.eval()

    print(f"model name: {args.model}")

    out = model(data.to(device))
    out_p = model(data_p.to(device))
    assert torch.allclose(out[backward_shuffle], out_p, atol=1e-14)


if __name__ == "__main__":
    test_perm_eq()
