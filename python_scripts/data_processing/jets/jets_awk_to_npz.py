import sys
from pathlib import Path

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"

sys.path.append(str(SCRIPT_PATH))


from data_processing.jets.npz_utils import (
    build_labels_array,
    build_input_array
)
from data_processing.jets.common_utils import (
    print_events,
    calculate_max_sample_length,
)
from data_processing.jets.preprocessing_header import (
    AWK_SAVE_LOC,
    NPZ_SAVE_LOC,
    NUM_CHUNK_THREADS,
    ENERGY_SCALE,
    OVERWRITE_NPZ
)
import awkward as ak
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import os
import time
from tqdm.auto import tqdm
from multiprocessing import Pool


DATA_FOLDERS = ["train", "val", "test"]


def read_parquet(filename):
    table = pq.read_table(filename)
    ak_array = ak.from_arrow(table)
    return ak_array


def build_arrays(data_folder_path, chunk_file_name):

    if not OVERWRITE_NPZ:
        print(f"Testing for existence of {os.path.join(NPZ_SAVE_LOC, data_folder_path.split('/')[-1], chunk_file_name + '.npz')}")
    if not OVERWRITE_NPZ and os.path.exists(os.path.join(NPZ_SAVE_LOC, data_folder_path.split('/')[-1], chunk_file_name + ".npz")):
        print(f"Already converted, skipping: {chunk_file_name}")
        return
    
    # /home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/attempt_1_june_18/full_set/SavedNpz/deltaR=0.2/energy_scale=1/train/user.mswiatlo.39955678._000005.mltree.root_chunk_0_train.parquet
    # /home/jhimmens/workspace/jetpointnet/pnet_data/processed_files/attempt_1_june_18/full_set/SavedNpz/deltaR=0.2/energy_scale=1/train/user.mswiatlo.39955678._000008.mltree.root_chunk_0_train.parquet.npz

    ak_array = read_parquet(os.path.join(data_folder_path, chunk_file_name))

    frac_labels = build_labels_array(
        ak_array, global_max_sample_length, "Fraction_Label"
    )
    tot_labels = build_labels_array(ak_array, global_max_sample_length, "Total_Label")
    tot_truth_e = build_labels_array(
        ak_array, global_max_sample_length, "Total_Truth_Energy"
    )

    # NOTE: energy_scale affects only cells energy; set to 1 to maintain same scale for track hits and cells
    feats = build_input_array(
        ak_array, global_max_sample_length, energy_scale=ENERGY_SCALE
    )

    # Save the feats and labels arrays to an NPZ file for each chunk
    npz_save_path = os.path.join(
        npz_data_folder_path, f"{chunk_file_name}.npz"
    )
    np.savez(
        npz_save_path,
        feats=feats,
        frac_labels=frac_labels,
        tot_labels=tot_labels,
        tot_truth_e=tot_truth_e,
    )


def build_arrays_wrapper(args):
    return build_arrays(*args)


if __name__ == "__main__":

    # Make sure this happens after SAVE_LOC is defined and created if necessary
    for folder in DATA_FOLDERS:
        folder_path = os.path.join(AWK_SAVE_LOC, folder)
        os.makedirs(
            folder_path, exist_ok=True
        )  # This line ensures the AWK_SAVE_LOC directories exist

    global_max_sample_length = 900 #find_global_max_sample_length()
    # global_max_sample_length = 278  # placeholder for now
    print(f"{global_max_sample_length = }")

    start_time = time.time()
    for data_folder in DATA_FOLDERS:
        npz_data_folder_path = os.path.join(NPZ_SAVE_LOC, data_folder)
        os.makedirs(npz_data_folder_path, exist_ok=True)  # Ensure the directory exists
        print(f"Processing data for: {data_folder}")

        data_folder_path = os.path.join(AWK_SAVE_LOC, data_folder)
        chunk_files = [
            f
            for f in os.listdir(data_folder_path)
            if f.endswith(".parquet")
        ]
        num_chunks = len(chunk_files)

        with Pool(processes=NUM_CHUNK_THREADS) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        build_arrays_wrapper,
                        zip([data_folder_path] * num_chunks, sorted(chunk_files)),
                    ),
                    total=num_chunks,
                )
            )
        print(f"Completed processing data for: {data_folder}")

    end_time = time.time()
    print(f"Processing took: {(end_time - start_time):.2f} seconds")
