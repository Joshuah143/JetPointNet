"""
Brief description of the script's purpose.

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
from jets_training.jets_train import train, GPU_ID, USER
RUN_SWEEP = False

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
WANDB_PROJECT = "pointcloud"

# 2: Define the search space
sweep_configuration = {
    "program": "python_scripts/jets_training/jets_train.py",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val/f1_score"},
    "parameters": {
        "BATCH_SIZE": {
            "distribution": "q_log_uniform_values",
            "min": 32,
            "max": 1024,
        },
        "LOSS_FUNCTION": {"values": ["BCE", "FocalBCE"]},
        "LOSS_ENERGY_WEIGHTING": {"values": ["absolute", 
                                             "square", 
                                             "normalize", 
                                             "standardize", 
                                             "threshold", 
                                             "none"]},
        "LR_MAX": {"max": 0.2, "min": 0.001}, 
        "LR_MIN": {"max": 0.01, "min": 0.00001}, 
        "LR_RAMP_EP": {"max": 10, "min": 1}, 
        "LR_SUS_EP": {"max": 10, "min": 0},
        "LR_DECAY": {"max": 1.0, "min": 0.01}
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project=WANDB_PROJECT,
    # description="Learning rate and batch size sweep.",
)

# Do not delete, this is the command to run a parallel sweep:
# CUDA_VISIBLE_DEVICES=5 wandb agent -p pointcloud -e jetpointnet SWEEP_ID
if RUN_SWEEP:
    wandb.agent(sweep_id, function=train)
else:
    print("Copy one (or more) of the following into a terminal:")
    for i in range(6):
        match USER:
            case "luclissa":
                print(f"CUDA_VISIBLE_DEVICES={i} wandb agent -p pointcloud -e jetpointnet {sweep_id}")
            case "jhimmens":
                print(f"cd ~/workspace/jetpointnet && CUDA_VISIBLE_DEVICES={i} wandb agent -p pointcloud -e jetpointnet {sweep_id}")

