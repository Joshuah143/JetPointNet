from multiprocessing import Pool
import os
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import awkward as ak
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

from data_processing.jets.common_utils import calculate_max_sample_length_simplified
from data_processing.jets.preprocessing_header import AWK_SAVE_LOC, SAMPLE_LENGTH_WORKERS, LEN


DATA_FOLDERS = ["train", "val", "test"]


def read_parquet(filename):
    table = pq.read_table(filename)
    ak_array = ak.from_arrow(table)
    return ak_array


def max_length_calculator_wrapper(full_path):
    ak_array = read_parquet(full_path)
    max_sample_length_arr, number_of_cells_in_track_arr, number_of_adj_track_arr, number_non_neg_adj_tracks = calculate_max_sample_length_simplified(ak_array)
    return max_sample_length_arr, full_path.split('/')[-1], number_of_cells_in_track_arr, number_of_adj_track_arr, number_non_neg_adj_tracks


def find_global_max_sample_length(n_files_per_set=-1):
    # hits_count = np.empty(shape=(0, 5))
    results = {}
    with Pool(processes=SAMPLE_LENGTH_WORKERS) as pool:
        for folder in tqdm(DATA_FOLDERS, desc="split loop"):
            data_sets = os.listdir(os.path.join(AWK_SAVE_LOC(LEN), folder))
            results[folder] = []
            for data_set in data_sets:
                print(data_set)
                folder_path = os.path.join(AWK_SAVE_LOC(LEN), folder, data_set, '*.parquet')
                files = [file for file in glob.glob(folder_path, recursive=True) if file.endswith(".parquet") and os.path.isfile(file)]
                files = files[:n_files_per_set]
                print(files)
                results[folder].extend(list(tqdm(pool.imap(max_length_calculator_wrapper, files),
                                            total=len(files),
                                            desc=f"Processing {folder}",
                                            leave=False)))
    
    metadata_path = Path(AWK_SAVE_LOC(LEN)) / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    with open(metadata_path / f"sample_length_calcs.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved to: {metadata_path / 'sample_length_calcs.json'}")


if __name__ == "__main__":
    print(f"Finding sample length")
    find_global_max_sample_length()
    print("Completed!")
