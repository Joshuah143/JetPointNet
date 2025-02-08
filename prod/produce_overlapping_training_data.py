from multiprocessing.pool import Pool
from pathlib import Path

import awkward as ak
import numpy as np
from tqdm.auto import tqdm
from loguru import logger as log

from .load_from_root_file import load_from_root
from .utils.data_loading import setup_directories, split_data
from .utils.to_numpy import event_to_trainable


# Params


def save_train_data(*, config: dict):
    log.info("Processing overlapping data pipeline")

    input_data_dir: Path = Path(config["data_pipeline"]["root_files_dir"])
    save_location: Path = Path(config["data_pipeline"]["output_dir"])
    geo_file: Path = Path(config["global_params"]["geo_file_loc"])
    desired_sets: list[str] = config["data_pipeline"]["sets_to_process"]
    set_to_dir_name: dict[str, str] = config["data_pipeline"]["set_paths"]
    data_split: dict[str, float] = config["data_pipeline"]["splits"]
    chunk_size: int = config["data_pipeline"]["overlapping"]["chunk_size"]
    max_delta_r: float = config["data_pipeline"]["max_delta_r"]
    max_sample_length: int = config["global_params"]["max_sample_length"]

    setup_directories(
        save_location,
        desired_sets,
        data_split,
        run_id=config["global_params"]["run_id"],
    )
    save_location = save_location / config["global_params"]["run_id"]

    for set_name in desired_sets:
        log.info(f"Handling set: {set_name}")
        log.info(f"Loading data from: {input_data_dir/set_to_dir_name[set_name]}")

        root_tree: ak.Array = load_from_root(
            input_data_dir / set_to_dir_name[set_name] / "*.root", geo_file, debug=False
        )

        split_tree = split_data(root_tree, data_split)
        tasks = []
        for split_type_name, data in split_tree.items():  # ('test', uproot items)
            for start_idx in range(0, len(data), chunk_size):
                chunk = data[start_idx : start_idx + chunk_size]
                split_save_location = save_location / split_type_name / set_name
                tasks.append(
                    (
                        split_type_name,
                        chunk,
                        split_save_location,
                        start_idx,
                        max_delta_r,
                        max_sample_length,
                    )
                )

        with Pool() as pool:
            for _ in tqdm(pool.imap(_process_split, tasks), total=len(tasks)):
                pass


def _process_split(args: tuple[str, ak.Array, Path, int, float, int]):
    split_type, data, split_save_location, split_id, max_delta_r, max_sample_length = (
        args
    )

    trainable = []
    for event in data:
        for track in range(len(event["tracks"])):
            # TODO: apply quality cuts here from old notebook, min pT, min hits, etc
            # What should be done in production with these hits?
            trainable.append(
                event_to_trainable(
                    event,
                    focal_index=track,
                    delta_r_max=max_delta_r,
                    max_event_len=max_sample_length,
                )
            )

    np.save(split_save_location / f"overlapping_{split_id}", np.stack(trainable))
    log.info(
        f"Saved {len(trainable)} events to {split_save_location / f'overlapping_{split_id}.npy'}"
    )
