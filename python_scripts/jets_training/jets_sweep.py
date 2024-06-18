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
from jets_training.jets_train import train, experiment_configuration, GPU_ID


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
WANDB_PROJECT = "pointcloud"


def main():
    wandb.init(
        project=WANDB_PROJECT,
        config=experiment_configuration,
        # config={
        #     "dataset": EXPERIMENT_NAME,
        #     "split_seed": SPLIT_SEED,
        #     "tf_seed": seed,
        #     "delta_R": MAX_DISTANCE,
        #     "energy_scale": ENERGY_SCALE,
        #     "n_points_per_batch": MAX_SAMPLE_LENGTH,
        #     "batch_size": BATCH_SIZE,
        #     "n_epochs": EPOCHS,
        #     "early_stopping_patience": ES_PATIENCE,  # not used
        #     "output_activation": OUTPUT_ACTIVATION_FUNCTION,
        #     "accuracy_energy_weight_scheme": ACC_ENERGY_WEIGHTING,
        #     "loss_energy_weight_scheme": ACC_ENERGY_WEIGHTING,
        #     "min_hits_per_track": 25,
        #     "fractional_energy_cutoff": FRACTIONAL_ENERGY_CUTOFF,
        #     "lr_max": LR_MAX,
        #     "lr_min": LR_MIN,
        #     "lr_ramp_ep": LR_RAMP_EP,
        #     "lr_sus_ep": LR_SUS_EP,
        #     "lr_decay": LR_DECAY,
        # },
        job_type="training",
        # tags=["baseline"],
    )
    train(wandb.config)


# 2: Define the search space
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val/f1_score"},
    "parameters": {
        "LR": {"distribution": "uniform", "min": 0.0001, "max": 0.1},
        "BATCH_SIZE": {
            # "distribution": "q_log_uniform_values",
            # "min": 32,
            # "max": 512,
            "values": [32, 64, 128, 256, 512]
        },
    },
}

for hyperparam, value in experiment_configuration.items():
    if hyperparam not in sweep_configuration["parameters"].keys():
        sweep_configuration["parameters"][hyperparam] = {"value": value}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project=WANDB_PROJECT,
    # description="Learning rate and batch size sweep.",
)

wandb.agent(sweep_id, function=train, count=100)
