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
from jets_training.jets_train import train, GPU_ID


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
WANDB_PROJECT = "pointcloud"

# 2: Define the search space
sweep_configuration = {
    "program": "python_scripts/jets_training/jets_train.py",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val/f1_score"},
    "parameters": {
        "LR": {"distribution": "uniform", "min": 0.0001, "max": 0.1},
        "BATCH_SIZE": {
            "distribution": "q_log_uniform_values",
            "min": 32,
            "max": 512,
        },
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project=WANDB_PROJECT,
    # description="Learning rate and batch size sweep.",
)

# CUDA_VISIBLE_DEVICES=5 wandb agent -p pointcloud -e jetpointnet k26j2xrm
wandb.agent(sweep_id, function=train, count=100)
