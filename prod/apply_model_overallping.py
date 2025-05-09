"""
For each event, apply the model to every track.
Use the model output to construct a track-cell matrix where the values are the model output feeatures.
Then apply reductions to normalize by cell.
Convert cells to clusters based onn cell outputs.
"""

import awkward as ak

from .JetPointNet import JetPointNet
import uproot
import os
from pathlib import Path
import numpy as np
from utils.data_loading import setup_directories

try:
    import tensorflow.keras as keras
except ImportError:
    import keras


def main():
    model_path = Path("some path!")  # path to model weights TODO: configure in config
    data_path = Path("some path!")  # path to root files
    step_size = 1000
    output_dir = Path("some path!")  # path to save results

    # load model
    model = JetPointNet(  # TODO: configure in config
        model_version=2,
        num_points=100,
        num_features=9,
        num_classes=1,
        output_activation_function="sigmoid",
    )
    model.load_weights(model_path)

    # Configure batch processing
    # Define processing parameters
    batch_size = 1000  # Number of events per batch
    output_dir = "output"  # Directory to save results

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of input files
    data_splits_names = ["train", "val", "test"]
    desired_sets = ["rho", "delta"]
    setup_directories(output_dir, desired_sets, data_splits_names)

    # Process each file in batches
    for data_split_name in data_splits_names:
        for desired_set in desired_sets:
            file_path = data_path / data_split_name / desired_set / "*.root"
            for idx, trees in enumerate(
                uproot.iterate(file_path, library="ak", step_size=step_size)
            ):
                events = trees["Events"]
                results = apply_model_to_events(model, events)
                output_file = (
                    output_dir / data_split_name / desired_set / f"{idx}.parquet"
                )  # This might have to be a json file depending on compatibility with ak arrays
                results.to_parquet(output_file)


def apply_model_to_events(model: keras.Model, events: ak.Array) -> ak.Array:
    """
    Apply the model to the events.

    Args:
        model: The model to apply.
        events: The events to apply the model to.

    Returns:
        The results of the model applied to the events.
        Events with no tracks and no cells, all are part of `attributed`.
    """
    pass


def apply_model_to_single_event(model: keras.Model, event: ak.Array) -> ak.Array:
    """
    Apply the model to a single event.
    """
    pass


if __name__ == "__main__":
    main()
