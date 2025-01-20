from pathlib import Path

import awkward as ak


def setup_directories(save_location: Path, desired_sets: list, data_split: dict):
    save_location.mkdir(exist_ok=True)
    for split_type in data_split.keys():
        split_save_location = save_location / split_type
        split_save_location.mkdir(exist_ok=True)
        for set_name in desired_sets:
            set_save_location = split_save_location / set_name
            set_save_location.mkdir(exist_ok=True)


def split_data(tree: ak.Array, data_splits: dict):
    split_tree = {}
    current_idx = 0
    end_idx = len(tree)
    for split_type, split_ratio in data_splits.items():
        subset_term = int(split_ratio * end_idx) + current_idx
        split_tree[split_type] = tree[current_idx:subset_term]
        current_idx = subset_term
    return split_tree
