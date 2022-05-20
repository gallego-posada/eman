import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import argparse

import exp_utils
import test_tforms
import torch

import models
from experiments.faust_direct import FAUST
from gem_cnn.transform.gem_precomp import GemPrecomp
from spiralnet.utils import preprocess_spiral


def eq_main(args):

    loader_args = argparse.Namespace(**{"dataset": "FAUST", "seed": args.seed})
    faust_tforms = test_tforms.faust_eval_tforms(loader_args)
    faust_tforms = test_tforms.load_transformed_dataset(FAUST, faust_tforms)

    untformed_test_loader = faust_tforms[1]["loader"]
    # Drop training and test set from loaders
    faust_tforms = faust_tforms[2:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gem_precomp = GemPrecomp(n_rings=args.n_rings, max_order=args.max_order)

    args.num_nodes = 6890
    args.target_dim = args.num_nodes

    # Create all models
    models = {}

    args.model = "GemCNN"
    args.equiv_bias = False
    models["GemCNN-non-equiv"] = create_model(args, device, aux_loader=None)

    args.model = "GemCNN"
    args.equiv_bias = True
    models["GemCNN-equiv"] = create_model(args, device, aux_loader=None)

    args.model = "RelTanGemCNN"
    args.equiv_bias = True
    models["RelTanGemCNN"] = create_model(args, device, aux_loader=None)

    args.model = "SpiralNet"
    models["SpiralNet"] = create_model(args, device, aux_loader=untformed_test_loader)

    target = exp_utils.construct_target("segmentation", args, device)
    prepare_batch = exp_utils.core_prepare_batch(target)

    def compare_outputs(model, batch, tform_batch, tform_dict):
        # out = model(inputs)
        if "shuffle" in tform_dict:
            shuffle = torch.tensor(tform_dict["shuffle"])
            backward_shuffle = test_tforms.shuffle_backwards(tform_dict["shuffle"])

            prepare_batch = exp_utils.core_prepare_batch(target)
            inputs, targets = prepare_batch(batch, device=device)

            out = model(inputs)

            inputs.pos = inputs.pos[shuffle]

            # update spiral indices
            model.conv1.indices = (
                backward_shuffle[model.conv1.indices][shuffle].clone().detach()
            )
            model.conv2.indices = (
                backward_shuffle[model.conv2.indices][shuffle].clone().detach()
            )
            model.conv3.indices = (
                backward_shuffle[model.conv3.indices][shuffle].clone().detach()
            )
            tform_out = model(inputs)

            model.conv1.indices = (
                shuffle[model.conv1.indices][backward_shuffle].clone().detach()
            )
            model.conv2.indices = (
                shuffle[model.conv2.indices][backward_shuffle].clone().detach()
            )
            model.conv3.indices = (
                shuffle[model.conv3.indices][backward_shuffle].clone().detach()
            )

            return ((out[shuffle] - tform_out) ** 2).mean().item()
        else:
            prepare_batch = exp_utils.core_prepare_batch(target)
            inputs, targets = prepare_batch(batch, device=device)
            out = model(inputs)

            tform_inputs, _ = prepare_batch(tform_batch, device=device)
            tform_out = model(tform_inputs)

        return ((out - tform_out) ** 2).mean().item()

    # Evaluate all models
    with torch.no_grad():
        for model_name, model in models.items():
            model.eval()
            for tform_dict in faust_tforms:
                equiv_err, num_batches = 0, 0
                for batch, tform_batch in zip(
                    untformed_test_loader, tform_dict["loader"]
                ):
                    equiv_err += compare_outputs(model, batch, tform_batch, tform_dict)
                    num_batches += 1.0

                print(model_name, tform_dict["name"], equiv_err / num_batches)


def create_model(args, device, aux_loader):
    if args.model == "SpiralNet":
        args.lr = 3e-3  # SpiralNet shows poor performance with higher learning rate
        d = next(iter(aux_loader))
        spiral_indices = preprocess_spiral(d.face.T, args.spiral_net_seq_length).to(
            device
        )
        model = models.__dict__[args.model](args, spiral_indices).to(device)
    else:
        model = models.__dict__[args.model](args).to(device)

    return model


if __name__ == "__main__":

    parser = exp_utils.parse_arguments()
    args = parser.parse_args()

    # args.model_batch = None # For classification
    args.model_batch = 100_000  # For segmentation

    eq_main(args)
