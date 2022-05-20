"""
Extract information logged to wandb in order to plot/analyze.
WandB help: https://docs.wandb.ai/guides/track/public-api-guide
"""

import argparse

import pandas as pd
import torch
import wandb

import models


def get_metrics(
    filters,
    metric_keys,
    config_keys=None,
    x_axis="_step",
    task_name="faust",
    entity="ger__man",
):
    """
    Extract metric_keys from wandb runs given filters. Keep config_keys for reference

    Args:
        filters: {"$and": [{"config.run_group": "4 oct"}, {"config.angular_bias": True}]}
        metric_keys: metrics we want to download: ["val/acc", "val/loss"]
        config_keys: config elements to return: ["seed", "model_type"]
        x_axis: one of "_step" or "epoch"

    Returns:
        DataFrame list of metrics, config list
    """

    API = wandb.Api(overrides={"entity": entity})
    runs = API.runs(path=entity + "/" + task_name, filters=filters, order="-created_at")
    print("Number of runs:", len(runs))

    all_metrics = pd.DataFrame()
    for run in runs:
        # samples param: without replacement, if too large returns all.
        metrics = run.history(samples=9000, keys=metric_keys, x_axis=x_axis)

        config = run.config

        # # Do not keep the whole config, only config_keys if provided by user
        if config_keys is not None:
            filtered_config = {
                key: val for key, val in config.items() if key in config_keys
            }
        else:
            filtered_config = config

        for key, val in filtered_config.items():
            metrics.insert(0, key, str(val))

        all_metrics = all_metrics.append(metrics)

    return all_metrics


def get_models(filters, task_name="faust", entity="ger__man"):
    """
    Get the models of runs according to filters

    Args:
        filters = {"$and": [{"config.run_group": "4 oct"}, {"config.angular_bias": True}]}

    Returns:
        dict[run_name] = {'model':, 'config':}
    """

    API = wandb.Api(overrides={"entity": entity})
    runs = API.runs(path=entity + "/" + task_name, filters=filters, order="-created_at")
    print("Number of runs:", len(runs))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res_dict = {}
    for i, run in enumerate(runs):

        run_res = {}

        suffix = None
        if ("save_model" in run.config) and run.config["save_model"]:
            suffix = ".h5"
        elif ("save_model_dict" in run.config) and run.config["save_model_dict"]:
            suffix = ".pt"

        if suffix is not None:
            print("Run {}/{}".format(i, len(runs)))
            print("Loading model for run:", run.name)

            # Get last iterate model
            run.file("models/trained_model" + suffix).download(
                root="wandb/", replace=True
            )
            if suffix == ".pt":
                config = argparse.Namespace(**run.config)
                model = models.__dict__[config.model](config).to(DEVICE)
                state_dict = torch.load(
                    "wandb/models/trained_model.pt", map_location=DEVICE
                )
                model.load_state_dict(state_dict)
            elif suffix == ".h5":
                model = torch.load("wandb/models/trained_model.h5", map_location=DEVICE)

            run_res["model"] = model
            run_res["config"] = run.config

        res_dict[run.name] = run_res

    return res_dict


if __name__ == "__main__":

    filter_dict = {"model": "GemCNN", "wb_tags": ["faust_model_save"]}
    api_filter = {
        "$and": [{"config." + key: val} for (key, val) in filter_dict.items()]
    }
    print(api_filter)

    # Replace line below with your WandB setting
    # ENTITY_NAME = CUSTOM_ENTITY_NAME

    res_dict = get_models(api_filter, entity=ENTITY_NAME, task_name="faust")

    print(res_dict)
