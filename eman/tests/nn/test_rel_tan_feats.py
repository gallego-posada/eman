# sys.path.append(".")
import argparse
import os.path as osp
import sys

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader

import experiments.exp_utils as exp_utils
from eman.transform.rel_tan_features import RelTanTransform
from eman.transform.roto_translation_transformer import (
    RotoTranslationTransformer,
    rotate_frame,
)
from experiments.faust_direct import FAUST
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh


def rel_tan_test(args):
    transform = T.Compose([compute_normals_edges_from_mesh, SimpleGeometry()])
    rot_trans_transform = RotoTranslationTransformer(translation_mag=0)
    path = exp_utils.get_dataset_path("faust_direct")
    FAUST.processed_dir = osp.join(path, "processed_direct")
    train_dataset = FAUST(
        path, train=True, pre_transform_train=transform, pre_transform_test=transform
    )

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    mesh = next(iter(dataloader))

    if args.rel_tan_test:
        rel_transform = RelTanTransform(args)
        rel_transform(mesh)
        rel_tang_feat1 = mesh.rel_tang_feat.clone()

        simple_geo = SimpleGeometry()
        mesh2 = Data(pos=mesh.pos, face=mesh.face)
        mesh2 = rot_trans_transform(mesh2.clone())
        mesh2 = compute_normals_edges_from_mesh(mesh2)
        mesh2 = simple_geo(mesh2)
        rel_transform(mesh2)
        rel_tang_feat2 = mesh2.rel_tang_feat.clone()
        assert torch.allclose(rel_tang_feat2, rel_tang_feat1, atol=1e-2)

    if args.frame_test:
        # TODO: fix this test
        frame1 = mesh.frame

        simple_geo = SimpleGeometry()
        mesh2 = Data(pos=mesh.pos, face=mesh.face)
        mesh2 = rot_trans_transform(mesh2.clone())
        mesh2 = compute_normals_edges_from_mesh(mesh2)
        mesh2 = simple_geo(mesh2)
        rel_transform(mesh2)
        frame2 = mesh2.frame
        frame1_rot = rotate_frame(
            frame=frame1,
            rot_angle=rot_trans_transform.rot_angle,
            rot_direction=rot_trans_transform.rot_direction,
        )
        assert torch.allclose(frame1_rot, frame2, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMAN Parser")

    parser.add_argument("-yaml", "--yaml_file", default="", type=str)

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-rel_tan_test", action="store_false")
    parser.add_argument("-frame_test", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--model", default="RelTanGemCNN", type=str)
    parser.add_argument("--n_rings", default=2, type=int)
    parser.add_argument("--n_heads", default=1, type=int)
    parser.add_argument("--max_order", default=2, type=int)
    parser.add_argument("--hid_dim", default=256, type=int)

    parser.add_argument("--spiral_net_input_channels", default=3, type=int)
    parser.add_argument("--spiral_net_num_classes", default=6890, type=int)
    parser.add_argument("--spiral_net_seq_length", type=int, default=10)

    parser.add_argument(
        "--rel_power_const",
        default=1.0,
        type=float,
        help="constant used as RelTan coefficients",
    )
    parser.add_argument(
        "--deg_power",
        default=1.5,
        type=float,
        help="Power of degree normalization in RelTan models",
    )
    parser.add_argument(
        "--rel_power_list",
        type=float,
        nargs="+",
        default=[0.7],
        help="list of powers of norm in RelTan models",
    )
    parser.add_argument(
        "-null_isolated",
        action="store_true",
        help="do zero-out features for isolated nodes?",
    )

    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate")
    parser.add_argument(
        "-gas",
        "--grad_accum_steps",
        default=1,
        type=int,
        help="apply optimizer step every several minibatches",
    )
    parser.add_argument(
        "-bs", "--batch_size", default=1, type=int, help="Number of meshes per batch"
    )
    parser.add_argument(
        "-tf",
        "--train_fraction",
        default=1,
        type=float,
        help="Fraction of training samples to use",
    )

    parser.add_argument(
        "-random_test_gauge", action="store_true", help="random gauge for testset"
    )
    parser.add_argument(
        "-equiv_bias", action="store_true", help="adds bias only to scalar terms"
    )
    parser.add_argument(
        "-regular_non_lin",
        action="store_true",
        help="uses regular non-linearity in residual blocks",
    )

    parser.add_argument("-roto_translation", action="store_true")

    parser.add_argument("-wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wb_tags", nargs="+", default=[])
    parser.add_argument("-wandb_offline", dest="use_wandb_offline", action="store_true")

    parser.add_argument("-save_model", action="store_true")
    parser.add_argument("-save_model_dict", action="store_true")

    args = parser.parse_args()
    rel_tan_test(args)
