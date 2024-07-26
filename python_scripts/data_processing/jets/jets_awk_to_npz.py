import awkward as ak
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import os
import time
from tqdm.auto import tqdm
from multiprocessing import Pool
import sys
import glob
from pathlib import Path

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"

sys.path.append(str(SCRIPT_PATH))


from data_processing.jets.npz_utils import (
    build_input_array
)
from data_processing.jets.preprocessing_header import *


DATA_FOLDERS = ["train", "val", "test"]
global_max_sample_length = MAX_SAMPLE_LENGTH


def read_parquet(filename):
    table = pq.read_table(filename)
    ak_array = ak.from_arrow(table)
    return ak_array


def build_arrays(data_folder, chunk_file_path, npz_data_folder_path):
    data_set_name = prefix_to_set[chunk_file_path.split("/")[-1][:FILE_PREFIX_LEN]]

    if not OVERWRITE_NPZ:
        print(f"Testing for existence of {os.path.join(NPZ_SAVE_LOC(NPZ), data_folder, data_set_name, chunk_file_path.split('/')[-1] + '.npz')}")
        if os.path.exists(os.path.join(NPZ_SAVE_LOC(NPZ), data_folder, data_set_name, chunk_file_path.split('/')[-1] + '.npz')):
            print(f"Already converted, skipping: {chunk_file_path}")
            return
    
    ak_array = read_parquet(chunk_file_path)

    # NOTE: energy_scale affects only cells energy; set to 1 to maintain same scale for track hits and cells
    feats, cell_truths = build_input_array(
        ak_array, global_max_sample_length, energy_scale=ENERGY_SCALE
    )

    os.makedirs(os.path.join(npz_data_folder_path, data_set_name), exist_ok=True)
    # Save the feats and labels arrays to an NPZ file for each chunk
    npz_save_path = os.path.join(
        npz_data_folder_path, data_set_name, f"{chunk_file_path.split('/')[-1]}.npz"
    )
    np.savez(
        npz_save_path,
        feats=feats,
        cell_truths=cell_truths
    )


def build_arrays_wrapper(args):
    return build_arrays(*args)


def main():
    # Make sure this happens after SAVE_LOC is defined and created if necessary
    for folder in DATA_FOLDERS:
        folder_path = os.path.join(AWK_SAVE_LOC(NPZ), folder)
        os.makedirs(
            folder_path, exist_ok=True
        )  # This line ensures the AWK_SAVE_LOC(NPZ) directories exist

    start_time = time.time()
    for data_folder in DATA_FOLDERS:
        npz_data_folder_path = os.path.join(NPZ_SAVE_LOC(NPZ), data_folder)
        os.makedirs(npz_data_folder_path, exist_ok=True)  # Ensure the directory exists
        print(f"Processing data for: {data_folder}")

        data_folder_path = os.path.join(AWK_SAVE_LOC(NPZ), data_folder, "**/*.parquet")
        chunk_files = [
            f
            for f in glob.glob(data_folder_path, recursive=True)
            if f.endswith(".parquet") and any(f.split("/")[-1].startswith(i) for i in NPZ_ALLOWED_PREFIXES) and os.path.isfile(f)
        ]
    
        num_chunks = len(chunk_files)

        with Pool(processes=NPZ_NUM_CHUNK_THREADS) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        build_arrays_wrapper,
                        zip([data_folder] * num_chunks, sorted(chunk_files), [npz_data_folder_path] * num_chunks),
                    ),
                    total=num_chunks,
                )
            )
            
        print(f"Completed processing data for: {data_folder}")

    end_time = time.time()
    print(f"Processing took: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()
