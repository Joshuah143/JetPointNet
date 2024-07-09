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
    max_sample_length_arr, number_of_cells_in_track_arr, number_of_adj_track_arr = calculate_max_sample_length_simplified(ak_array)
    # print("Max sample length found: ", max_sample_length)
    return max_sample_length_arr, full_path.split('/')[-1], number_of_cells_in_track_arr, number_of_adj_track_arr


def find_global_max_sample_length():
    # hits_count = np.empty(shape=(0, 5))
    results = {}
    with Pool(processes=SAMPLE_LENGTH_WORKERS) as pool:
        for folder in tqdm(DATA_FOLDERS, desc="split loop"):
            folder_path = os.path.join(AWK_SAVE_LOC(LEN), folder, '**/*.parquet')
            files = [file for file in glob.glob(folder_path, recursive=True) if file.endswith(".parquet") and os.path.isfile(file)]
            results[folder] = list(tqdm(pool.imap(max_length_calculator_wrapper, files),
                                        total=len(files),
                                        desc=f"Processing {folder}",
                                        leave=False))
    
    metadata_path = Path(AWK_SAVE_LOC(LEN)) / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    with open(metadata_path / f"sample_length_calcs.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    print(f"Finding sample length")
    find_global_max_sample_length()
    print("Completed!")


"""" old in case of rollback for stats


 print(f"Global Max Sample Length: {global_max_sample_length}")
    hits_df = pd.DataFrame(
        hits_count,
        columns=["eventNumber", "trackID", "nHits", "nCell", "nUnfocusHits"],
        dtype=int,
    )
    hits_df.sort_values(["eventNumber", "trackID"], inplace=True)
    # hits_df["nTrack"] = hits_df.groupby(["eventNumber"]).trackID.count().values # L: don't work due to repeated eventNumbers, would need join

    # dump metadata
    metadata_path = Path(folder_path).parent / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    hits_df.to_csv(metadata_path / f"hits_per_event.csv", index=False)

    return global_max_sample_length

"""