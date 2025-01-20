from multiprocessing.pool import Pool
from pathlib import Path

import awkward as ak
import numpy as np
from tqdm.auto import tqdm

from load_from_root_file import load_from_root
from utils.data_loading import setup_directories, split_data
from utils.dev_tools import load_config
from utils.to_numpy import event_to_trainable

config = load_config()

# Params
input_data_dir = Path(config["data_pipeline"]["root_files_dir"])
output_data_dir = Path(config["data_pipeline"]["output_dir"])
geo_file = Path(config["global_params"]["geo_file_loc"])

MAX_DELTA_R = config["data_pipeline"]["max_delta_r"]
max_sample_length = config["global_params"]["max_sample_length"]
desired_sets = config["data_pipeline"]["sets_to_process"]
set_to_dir_name = config["data_pipeline"]["set_paths"]
data_split = config["data_pipeline"]["splits"]
save_location = Path(config["data_pipeline"]["output_dir"])
chunk_size = config["data_pipeline"]["overlapping"]["chunk_size"]


def save_train_data(chunk_size: int = 1):
    setup_directories(save_location, desired_sets, data_split)
    # TODO: should warn if the output directory already exists as data may not be overwritten causing issues
    for set_name in desired_sets:
        print(f"Handling set: {set_name}")
        print(f"Loading data from: {input_data_dir/set_to_dir_name[set_name]}")

        root_tree: ak.Array = load_from_root(
            input_data_dir / set_to_dir_name[set_name] / "*.root", geo_file, debug=False
        )

        split_tree = split_data(root_tree, data_split)
        tasks = []
        for split_type_name, data in split_tree.items():  # ('test', uproot items)
            for start_idx in range(0, len(data), chunk_size):
                chunk = data[start_idx : start_idx + chunk_size]
                split_save_location = save_location / split_type_name / set_name
                tasks.append((split_type_name, chunk, split_save_location, start_idx))
        print(tasks)
        with Pool() as pool:
            for _ in tqdm(pool.imap(_process_split, tasks), total=len(tasks)):
                pass


def _process_split(args):
    split_type, data, split_save_location, split_id = args

    trainable = []
    for event in data:
        for track in range(len(event["tracks"])):
            # TODO: apply quality cuts here from old notebook, min pT, min hits, etc
            # What should be done in production with these hits?
            trainable.append(event_to_trainable(event, focal_index=track))

    np.save(split_save_location / f"overlapping_{split_id}.npz", np.stack(trainable))
    print(
        f"Saved {len(trainable)} events to {split_save_location / f'overlapping_{split_id}.npz'}"
    )
