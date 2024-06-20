from multiprocessing import Pool
import os
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import awkward as ak
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

from data_processing.jets.common_utils import calculate_max_sample_length
from data_processing.jets.preprocessing_header import AWK_SAVE_LOC, SAMPLE_LENGTH_WORKERS


DATA_FOLDERS = ["train", "val", "test"]

def read_parquet(filename):
    table = pq.read_table(filename)
    ak_array = ak.from_arrow(table)
    return ak_array


def max_length_calculator_wrapper(full_path):
    ak_array = read_parquet(full_path)
    max_sample_length, n_points = calculate_max_sample_length(ak_array)
    # print("Max sample length found: ", max_sample_length)
    return max_sample_length, n_points, full_path.split('/')[-1]


def find_global_max_sample_length():
    # hits_count = np.empty(shape=(0, 5))
    results = {}
    with Pool(processes=SAMPLE_LENGTH_WORKERS) as pool:
        for folder in tqdm(DATA_FOLDERS, desc="split loop"):
            folder_path = os.path.join(AWK_SAVE_LOC, folder)
            files = [folder_path + '/' + file for file in os.listdir(folder_path) if file.endswith(".parquet")]
            results[folder] = list(tqdm(pool.imap(max_length_calculator_wrapper, files),
                                        total=len(files),
                                        desc=f"Processing {folder}",
                                        leave=False))
            results[folder] = [[max_sample_length, n_points.tolist(), chunk_name] for max_sample_length, n_points, chunk_name in results[folder]]
    
    metadata_path = Path(AWK_SAVE_LOC).parent / "metadata"
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