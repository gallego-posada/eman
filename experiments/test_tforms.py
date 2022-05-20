import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


import argparse
import os.path as osp
import pickle

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from faust_direct import FAUST
from ignite.engine import create_supervised_evaluator
from torch_geometric.data import DataLoader
from torch_geometric.transforms import FaceToEdge
from tosca_direct import TOSCA

import experiments.exp_utils as exp_utils
from eman.transform.permute_nodes import PermuteNodes
from eman.transform.roto_translation_transformer import RotoTranslationTransformer
from experiments.exp_utils import set_seed
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_table(
    tforms, res_dict=None, res_dict_path=None, equiv_comp=False, give_std=True
):

    if res_dict is None:
        with open(res_dict_path, "rb") as handle:
            res_dict = pickle.load(handle)

    # Load res_dict into dataframe. Expand config and test internal dictionaries
    df = pd.DataFrame.from_dict(res_dict).transpose()
    exp_conf = pd.DataFrame(df["config"].tolist(), index=df.index)
    df = pd.concat([df.drop("config", axis=1), exp_conf], axis=1)

    tform_names = [_["name"] for _ in tforms]
    df_list = []
    for _ in tform_names:
        a = pd.DataFrame(df[_].tolist(), index=df.index)
        df_list.append(a.add_prefix(_ + "-"))

    df = pd.concat([df] + df_list, axis=1)
    df = df.drop(tform_names, axis=1)
    df["model"] = df["model"] + " - " + df["rel_power_list"].astype(str)

    if equiv_comp:
        df["model"] = df["model"] + " - EqBias: " + df["equiv_bias"].astype(str)

    # Aggregate (over the seeds) based on model name
    agg_params = {}  # {_+'-nll': ['median'] for _ in tform_names}
    agg_metrics = ["median", "std"] if give_std else ["median"]
    agg_params.update({_ + "-accuracy": agg_metrics for _ in tform_names})
    agg_df = df.groupby("model").agg(agg_params)

    df2 = 100 * agg_df.sort_values(("test-accuracy", "median"))
    df2 = df2.round(2)

    return df2


def shuffle_backwards(shuffle):
    a = torch.arange(len(shuffle))
    a[shuffle] = torch.arange(len(shuffle))
    return a


def create_transformed_loader(CLS, root, processed_path, tform, name, train):

    CLS.processed_dir = osp.join(root, processed_path)
    dataset = CLS(root, train=train, pre_transform=tform, pre_transform_str=name)
    path = dataset.processed_paths[train]

    orig_name = name
    name = name + "_" if name != "" else name
    name = name + "train" if train else name + "test"

    return {
        "orig_name": orig_name,
        "name": name,
        "train": train,
        "path": path,
        "root": root,
        "processed_path": processed_path,
    }


def load_transformed_dataset(CLS, tforms, shuffle=False):

    for tform_dict in tforms:
        CLS.processed_dir = osp.join(tform_dict["root"], tform_dict["processed_path"])
        # Explicitly skip process
        dataset = CLS(
            tform_dict["root"],
            train=tform_dict["train"],
            pre_transform=None,
            pre_transform_str=tform_dict["orig_name"],
            skip_process=True,
        )
        # Explicitly load stored file
        dataset.data, dataset.slices = torch.load(tform_dict["path"])
        tform_dict["loader"] = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    return tforms


def faust_eval_tforms(args):

    set_seed(args.seed)

    root = exp_utils.get_dataset_path("faust_direct")

    tforms = []

    # Evaluate on train and test sets with train transform for sanity check
    train_tform = T.Compose([compute_normals_edges_from_mesh, SimpleGeometry()])
    tforms.append(
        create_transformed_loader(
            FAUST, root, "processed_direct", train_tform, name="", train=True
        )
    )
    tforms.append(
        create_transformed_loader(
            FAUST, root, "processed_direct", train_tform, name="", train=False
        )
    )

    # NOTE: If you experience unexpected results, try removing the "process_equiv" folder

    # Random gauge
    tforms.append(
        create_transformed_loader(
            FAUST,
            root,
            "processed_equiv",
            T.Compose(
                [compute_normals_edges_from_mesh, SimpleGeometry(gauge_def="random")]
            ),
            name="gauge",
            train=False,
        )
    )

    # Rotations
    tforms.append(
        create_transformed_loader(
            FAUST,
            root,
            "processed_equiv",
            T.Compose(
                [
                    RotoTranslationTransformer(translation_mag=10),
                    compute_normals_edges_from_mesh,
                    SimpleGeometry(),
                ]
            ),
            name="roto",
            train=False,
        )
    )

    # Permutations
    rng = np.random.default_rng(args.seed)
    shuffle = rng.choice(6890, size=6890, replace=False)
    tforms.append(
        create_transformed_loader(
            FAUST,
            root,
            "processed_equiv",
            T.Compose(
                [
                    PermuteNodes(shuffle),
                    compute_normals_edges_from_mesh,
                    SimpleGeometry(),
                ]
            ),
            name="perm",
            train=False,
        )
    )
    tforms[-1].update({"shuffle": shuffle})

    return tforms


def tosca_eval_tforms(args):

    set_seed(args.seed)

    root = exp_utils.get_dataset_path("tosca_direct")

    tforms = []

    # Evaluate on train and test sets with train transform for sanity check
    train_tform = T.Compose(
        [
            FaceToEdge(remove_faces=False),
            compute_normals_edges_from_mesh,
            SimpleGeometry(),
        ]
    )
    tforms.append(
        create_transformed_loader(
            TOSCA, root, "processed_direct", train_tform, name="", train=True
        )
    )
    tforms.append(
        create_transformed_loader(
            TOSCA, root, "processed_direct", train_tform, name="", train=False
        )
    )

    # NOTE: If you experience unexpected results, try removing the "process_equiv" folder

    # Random gauge
    tforms.append(
        create_transformed_loader(
            TOSCA,
            root,
            "processed_equiv",
            T.Compose(
                [
                    FaceToEdge(remove_faces=False),
                    compute_normals_edges_from_mesh,
                    SimpleGeometry(gauge_def="random"),
                ]
            ),
            name="gauge",
            train=False,
        )
    )

    # Rotations
    tforms.append(
        create_transformed_loader(
            TOSCA,
            root,
            "processed_equiv",
            T.Compose(
                [
                    RotoTranslationTransformer(translation_mag=100),
                    FaceToEdge(remove_faces=False),
                    compute_normals_edges_from_mesh,
                    SimpleGeometry(),
                ]
            ),
            name="roto",
            train=False,
        )
    )

    # Not using permutations in TOSCA -> meshes have different # of nodes

    return tforms


def run_evaluations(runs_dict, tforms, metrics_dict, dataset, verbose=False):

    eval_res = {}

    for run_name, run_res in runs_dict.items():
        if "model" in run_res.keys():
            model_res = {}

            model_res["config"] = run_res["config"]
            config = argparse.Namespace(**run_res["config"])

            model = run_res["model"]
            model.eval()

            if verbose:
                print("Model for run: ", run_name)
            for tform_dict in tforms:

                task_type = "segmentation" if dataset == "FAUST" else "classification"
                target = exp_utils.construct_target(task_type, config, DEVICE)
                if "shuffle" in tform_dict.keys():
                    target = target[shuffle_backwards(tform_dict["shuffle"])]

                prepare_batch_fn = exp_utils.core_prepare_batch(target)
                evaluator = create_supervised_evaluator(
                    model,
                    metrics=metrics_dict,
                    device=DEVICE,
                    prepare_batch=prepare_batch_fn,
                )

                evaluator.run(tform_dict["loader"])
                metrics = evaluator.state.metrics
                if verbose:
                    print(
                        f"{tform_dict['name'].upper()} Results "
                        f"Avg accuracy: {metrics['accuracy']:.5f} Avg loss: {metrics['nll']:.5f}"
                    )

                model_res[tform_dict["name"]] = metrics

            eval_res[run_name] = model_res

    return eval_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMAN evaluation parser")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--dataset", default="FAUST", type=str)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.dataset == "FAUST":
        res_dict = eval_faust_models(args)
    elif args.dataset == "TOSCA":
        res_dict = eval_tosca_models(args)
    else:
        raise ValueError

    for run_name, run_dict in res_dict.items():
        for key, metrics in run_dict.items():
            if "config" != key:
                print(run_name, metrics)
