import glob
from pathlib import Path

import numpy as np

input_data_dir = Path("")
output_data_dir = Path("")
desired_sets = (
    # "rho",
    # "delta",
    # "JZ0",
    # "JZ1",
    # "JZ2",
    # "JZ3",
    "JZ4",
    #  "JZ5",
    #  "JZ6",
    #  "JZ7",
    #  "JZ8",
    #  "JZ9",
)

data_splits = (
    "train",
    "test",
    "val",
)

file_chunk_sizes = 1000  # number of events in a numpy file


def chunk_files():
    setup_directories(output_data_dir)
    for split in data_splits:
        for set in desired_sets:
            files_to_chunk = glob.glob(input_data_dir / split / set / "*.npz")
            buffer = np.array([])
            for file in files_to_chunk:
                # TODO: finish this method
                pass


def setup_directories(save_location: Path):
    save_location.mkdir(exist_ok=True)
    for split_type in data_splits:
        split_save_location = save_location / split_type
        split_save_location.mkdir(exist_ok=True)
        for set_name in desired_sets:
            set_save_location = split_save_location / set_name
            set_save_location.mkdir(exist_ok=True)


if __name__ == "__main__":
    chunk_files()
