import glob
from pathlib import Path

import numpy as np

from utils.data_loading import setup_directories
from utils.dev_tools import load_config

config = load_config()


def chunk_files(
    desired_sets: set | list,
    data_splits_names: set | list,
    input_data_dir: Path,
    output_data_dir: Path,
    file_chunk_sizes: int = 1000,
):
    setup_directories(output_data_dir, desired_sets, data_splits_names)
    for split in data_splits_names:
        for set_ in desired_sets:
            print(f"Chunking data for {split} {set_}")
            print(f"Loading data from: {input_data_dir / split / set_}")

            files_to_chunk = glob.glob(str(input_data_dir / split / set_ / "*.npy"))
            save_path = output_data_dir / split / set_
            if not files_to_chunk:
                continue  # Skip if no files found

            # TODO: this is memory intensive,
            #  consider sub-chunking in a more memory efficient way
            all_arrays = []
            for file_path in files_to_chunk:
                npz_file = np.load(file_path)
                all_arrays.append(npz_file)

            data = np.concatenate(all_arrays, axis=0)
            np.random.shuffle(data)

            num_rows = data.shape[0]
            start_idx = 0
            chunk_count = 0

            while start_idx < num_rows:
                end_idx = min(start_idx + file_chunk_sizes, num_rows)
                chunk_data = data[start_idx:end_idx]

                chunk_filename = f"{split}_{set_}_chunk_{chunk_count}.npz"
                np.savez_compressed(save_path / chunk_filename, chunk_data)

                start_idx = end_idx
                chunk_count += 1


if __name__ == "__main__" and config["data_chunking"]["enabled"]:
    chunk_files(
        desired_sets=config["data_chunking"]["enabled_sets"],
        data_splits_names=config["data_chunking"]["enabled_splits"],
        input_data_dir=Path(config["data_chunking"]["input_data_path"]),
        output_data_dir=Path(config["data_chunking"]["output_data_path"]),
        file_chunk_sizes=config["data_chunking"]["chunk_size"],
    )
