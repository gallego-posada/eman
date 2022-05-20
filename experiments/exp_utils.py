import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import argparse
import json
import os.path as osp
import random

import numpy as np
import torch
import wandb
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, RunningAverage

import experiments.models as models
from spiralnet.utils import preprocess_spiral


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_dataset_path(task_name: str) -> str:
    # Retrieve path for given task_name

    with open(osp.join(osp.dirname(__file__), "paths.json")) as f:
        all_paths = json.load(f)
        path = all_paths[task_name]

    return path


def construct_target(task_type="segmentation", args=None, device=None):
    if task_type == "segmentation":
        aux = torch.arange(args.num_nodes, dtype=torch.long, device=device)
        return aux.expand(args.batch_size, args.num_nodes).flatten()
    if task_type == "classification":
        return None

    raise ValueError("Task type {} not understood".format(task_type))


def core_prepare_batch(target=None, shuffle=None):
    if target is None:

        def prepare_batch(batch, device, non_blocking=False):
            data = batch.to(device)
            data.pos = data.pos.float()
            if shuffle is None:
                data.pos = data.pos.float()
            else:
                data.pos = data.pos.float()[shuffle]
            return data, data.y.to(device)

    else:

        def prepare_batch(batch, device, non_blocking=False):
            data = batch.to(device)
            if shuffle is None:
                data.pos = data.pos.float()
            else:
                data.pos = data.pos.float()[shuffle]
            return data, target.to(device)

    return prepare_batch


class GEMEngine:
    def __init__(
        self, model, optimizer, criterion, device, prepare_batch, grad_accum_steps=1
    ):

        self.trainer = create_supervised_trainer(
            model,
            optimizer,
            criterion,
            device=device,
            prepare_batch=prepare_batch,
            gradient_accumulation_steps=grad_accum_steps,
        )

        RunningAverage(output_transform=lambda x: x).attach(self.trainer, "loss")
        ProgressBar().attach(self.trainer, ["loss"])

        metrics_dict = {"nll": Loss(criterion), "accuracy": Accuracy()}

        self.eval_dict = {}
        for eval_name in ["train", "test", "test_tf"]:
            self.eval_dict[eval_name] = create_supervised_evaluator(
                model, metrics=metrics_dict, device=device, prepare_batch=prepare_batch
            )

    def set_epoch_loggers(self, loaders_dict):
        def inner_log(engine, tag):
            self.eval_dict[tag].run(loaders_dict[tag])
            metrics = self.eval_dict[tag].state.metrics
            print(
                f"{tag.upper()} Results - Epoch: {engine.state.epoch} "
                f"Avg accuracy: {metrics['accuracy']:.5f} Avg loss: {metrics['nll']:.5f}"
            )

        # These logs are created regardless of the wandb choice
        if loaders_dict["train"] is not None:

            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_train_results(engine):
                inner_log(engine, "train")

        if loaders_dict["test"] is not None:

            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_test_results(engine):
                inner_log(engine, "test")

    def create_wandb_logger(
        self, log_interval=1, entity="ger__man", task_name="faust", tags=[], config={}
    ):

        wandb_logger = WandBLogger(
            entity=entity, project=task_name, config=config, tags=tags
        )

        # Attach the logger to the trainer to log training loss at each iteration
        wandb_logger.attach_output_handler(
            self.trainer,
            event_name=Events.ITERATION_COMPLETED(every=log_interval),
            tag="train",
            output_transform=lambda loss: {"batchloss": loss},
            state_attributes=["epoch"],
        )

        for tag in ["train", "test"]:
            wandb_logger.attach_output_handler(
                self.eval_dict[tag],
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["nll", "accuracy"],
                global_step_transform=lambda *_: self.trainer.state.iteration,
            )

        return wandb_logger


class Experiment:
    def __init__(self, task_name, task_type, construct_loaders):
        self.task_name = task_name
        self.task_type = task_type
        self.construct_loaders = construct_loaders

    def main(self, args):

        set_seed(args.seed)

        # --------------- Dataset ---------------
        (
            transform,
            loaders_dict,
            args.num_nodes,
            args.target_dim,
        ) = self.construct_loaders(args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.task_type == "classification":
            args.model_batch = None
        if self.task_type == "segmentation":
            args.model_batch = 100_000

        if args.verbose:
            print(f"Building model of type: {args.model}")
        if args.model == "SpiralNet":
            args.lr = 3e-3  # SpiralNet shows poor performance with higher learning rate
            d = next(iter(loaders_dict["train"]))
            spiral_indices = preprocess_spiral(d.face.T, args.spiral_net_seq_length).to(
                device
            )
            model = models.__dict__[args.model](args, spiral_indices).to(device)
        else:
            model = models.__dict__[args.model](args).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.NLLLoss()

        # --------------- Train ---------------
        target = construct_target(self.task_type, args, device)
        prepare_batch_fn = core_prepare_batch(target)

        engine = GEMEngine(
            model,
            optimizer,
            criterion,
            device,
            prepare_batch_fn,
            grad_accum_steps=args.grad_accum_steps,
        )
        engine.set_epoch_loggers(loaders_dict)

        if args.use_wandb:
            if args.use_wandb_offline:
                os.environ["WANDB_MODE"] = "offline"

            wandb_logger = engine.create_wandb_logger(
                entity="ger__man",
                task_name=self.task_name,
                tags=args.wb_tags,
                config=vars(args),
            )

        engine.trainer.run(loaders_dict["train"], max_epochs=args.epochs)

        if args.save_model or args.save_model_dict:

            foldername = "model_checkpoints"
            folder_path = foldername
            if wandb.run is not None:
                folder_path = os.path.join(wandb.run.dir, foldername)
            os.makedirs(folder_path, exist_ok=True)

            if args.save_model:
                file_path = f"{folder_path}/trained_model.h5"
                torch.save(model, file_path)
            elif args.save_model_dict:
                file_path = f"{folder_path}/trained_model_dict.pt"
                torch.save(model.state_dict(), file_path)

            if wandb.run is not None:
                wandb.save(file_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="EMAN Parser")

    parser.add_argument("-yaml", "--yaml_file", default="", type=str)

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--model", default="GemCNN", type=str)
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

    return parser


def run_parser(parser):
    args = parser.parse_args()
    if args.yaml_file != "":
        opt = yaml.load(open(args.yaml_file), Loader=yaml.FullLoader)
        args.__dict__.update(opt)
    return args
