import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import glob
import os.path as osp
from typing import Callable, Optional

import exp_utils
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.io import read_txt_array
from torch_geometric.transforms import FaceToEdge

from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh


class TOSCA(torch_geometric.datasets.tosca.TOSCA):
    def __init__(
        self,
        root: str,
        train: bool = True,
        pre_transform: Optional[Callable] = None,
        pre_transform_str="",
        pre_filter: Optional[Callable] = None,
        train_frac: float = 0.75,
        skip_process: bool = False,
    ):

        self.pre_transform = pre_transform
        self.pre_transform_str = pre_transform_str
        self.train_frac = train_frac
        self.skip_process = skip_process
        super().__init__(root, pre_transform=pre_transform, pre_filter=pre_filter)

        path = self.processed_paths[train]
        if not self.skip_process:
            self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        name = "_".join([cat[:2] for cat in self.categories])
        base_paths = [f"{name}" + _ + ".pt" for _ in ["test", "train"]]
        return [self.pre_transform_str + _ for _ in base_paths]

    def _process_mesh(self, data, tform, dlist):
        aux_data = copy.deepcopy(data)
        if tform is not None:
            aux_data = tform(aux_data)
        dlist.append(aux_data)

    def process(self):
        if not self.skip_process:
            tr_dlist, ts_dlist = [], []

            for cat_id, cat in enumerate(self.categories):
                print(f"Category: {cat}")
                paths = glob.glob(osp.join(self.raw_dir, f"{cat}*.tri"))
                paths = [path[:-4] for path in paths]
                paths = sorted(paths, key=lambda e: (len(e), e))

                # Take the first (1 - self.train_frac)% meshes as test set
                tst_paths = paths[: int(len(paths) * (1 - self.train_frac))]

                for path in paths:
                    pos = read_txt_array(f"{path}.vert")
                    face = read_txt_array(f"{path}.tri", dtype=torch.long)
                    face = face - face.min()
                    data = Data(
                        pos=pos, face=face.t().contiguous(), y=torch.tensor(cat_id)
                    )
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if not (path in tst_paths):  # This is a training mesh
                        self._process_mesh(data, self.pre_transform, tr_dlist)
                    else:  # Mesh is in the test paths
                        self._process_mesh(data, self.pre_transform, ts_dlist)

            # Save paths = [test path, train path]
            for data_list, _path in zip([ts_dlist, tr_dlist], self.processed_paths):
                torch.save(self.collate(data_list), _path)


def construct_loaders(args):
    # --------------- Dataset transformations ---------------
    # Define transformations to be performed on the dataset:
    max_order = args.max_order

    # Number of rings in the radial profile
    n_rings = args.n_rings

    # Number of meshes per batch
    batch_size = args.batch_size

    # Transform mesh faces to graph using FaceToEdge
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/tosca.html#TOSCA
    # Transformation that computes a multi-scale radius graph and precomputes the logarithmic map.

    # make sure the pre_transform is applied before the processed data is generated
    pre_tform = T.Compose(
        [
            FaceToEdge(remove_faces=False),
            compute_normals_edges_from_mesh,
            SimpleGeometry(),
        ]
    )

    # Not needed here, called in model.py
    transform = GemPrecomp(n_rings, max_order)

    # --------------- Set dataset path ---------------
    # Provide a path to load and store the dataset.
    path = exp_utils.get_dataset_path("tosca_direct")
    # Monkey patch to change processed dir, to allow for direct and pool pre-processing
    TOSCA.processed_dir = osp.join(path, "processed_direct")

    # --------------- Load datasets ---------------
    if args.verbose:
        print("Creating datasets")
    train_dataset = TOSCA(root=path, train=True, pre_transform=pre_tform)

    # Filter a fraction of the dataset if required
    if args.train_fraction != 1:
        assert 0 < args.train_fraction <= 1
        num_samples = int(len(train_dataset) * args.train_fraction)
        print("Training using {} samples".format(num_samples))
        ix_choice = np.random.choice(len(train_dataset), num_samples, replace=False)
        train_data = [train_dataset.get(_) for _ in ix_choice]
    else:
        train_data = train_dataset

    # Scale batch_size according to train fraction
    train_batch_size = max(1, int(batch_size * args.train_fraction))
    num_nodes = train_dataset[0].num_nodes

    # Evaluation takes place on full test set
    # Unseen and NOT-transformed meshes
    test_dataset = TOSCA(root=path, train=False, pre_transform=pre_tform)

    loaders_dict = {
        "train": DataLoader(train_data, batch_size=train_batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }

    # There are 9 classes in this dataset
    target_dim = 9

    return transform, loaders_dict, num_nodes, target_dim


if __name__ == "__main__":

    experiment = exp_utils.Experiment(
        task_name="tosca",
        task_type="classification",
        construct_loaders=construct_loaders,
    )

    parser = exp_utils.parse_arguments()
    parser.add_argument(
        "-mean_pooling",
        action="store_true",
        help="perform avg pooling before FC layers",
    )

    try_args = exp_utils.run_parser(parser)
    try_args.mean_pooling = True
    try_args.null_isolated = True

    experiment.main(try_args)
