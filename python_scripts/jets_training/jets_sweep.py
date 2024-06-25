"""
This script runs a wandb sweep. The idea is to import the `train` function and a baseline `experiment_configuration` from jets_train.py and use them for the sweep.

The sweep parameters are defined here ith `sweep_configuration` dict. The missing ones are given default values based on `experiment_configuration`.

Author: Luca Clissa <luca.clissa2@unibo.it>
Created: 2024-06-17
License: Apache License 2.0
"""

import sys
from pathlib import Path

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"

sys.path.append(str(SCRIPT_PATH))

import os
import wandb
from jets_training.jets_train import train, baseline_configuration


WANDB_PROJECT = "pointcloud"

# 2: Define the search space
sweep_configuration = {
    "program": "python_scripts/jets_training/jets_train.py",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val/f1_score"},
    "parameters": {
        "LR": {"distribution": "uniform", "min": 0.001, "max": 0.1},
        "BATCH_SIZE": {
            # "distribution": "q_log_uniform_values",
            # "min": 32,
            # "max": 512,
            "values": [256, 512]
        },
        # "LOSS_FUNCTION": {"values": ["BinaryCrossentropy", "BinaryFocalCrossentropy"]},
        # "LOSS_ENERGY_WEIGHTING": {
        #     "values": [
        #         "absolute",
        #         "square",
        #         "normalize",
        #         "standardize",
        #         "threshold",
        #         "none",
        #     ]
        # },
        # "LR_MAX": {"max": 0.2, "min": 0.001},
        # "LR_MIN": {"max": 0.01, "min": 0.00001},
        # "LR_RAMP_EP": {"max": 10, "min": 1},
        # "LR_SUS_EP": {"max": 10, "min": 0},
        # "LR_DECAY": {"max": 1.0, "min": 0.01},
    },
}


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=WANDB_PROJECT,
    )
    print("Copy one (or more) of the following into a terminal to start the sweep:")
    for i in range(6):
        print(f"cd ~/workspace/jetpointnet && CUDA_VISIBLE_DEVICES={i} wandb agent -p pointcloud -e jetpointnet {sweep_id}")


    #wandb.agent(sweep_id, function=train, count=200)
