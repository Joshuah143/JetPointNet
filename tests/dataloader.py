import sys
from pathlib import Path
import os

REPO_PATH = Path.home() / "workspace/jetpointnet"
SCRIPT_PATH = REPO_PATH / "python_scripts/data_processing/jets"
sys.path.append(str(SCRIPT_PATH))
SCRIPT_PATH = REPO_PATH / "python_scripts"
sys.path.append(str(SCRIPT_PATH))

import numpy as np
import glob
from data_processing.jets.preprocessing_header import MAX_DISTANCE


# SET PATHS FOR I/O AND CONFIG
USER = Path.home().name
if USER == "jhimmens":
    OUTPUT_DIRECTORY_NAME = "2000_events_w_fixed_hits"
    DATASET_NAME = "raw"
    GPU_ID = "1"
elif USER == "luclissa":
    OUTPUT_DIRECTORY_NAME = "ttbar"
    DATASET_NAME = "benchmark"
    GPU_ID = "0"
else:
    raise Exception("UNKOWN USER")

ENERGY_SCALE = 1000
EXPERIMENT_NAME = f"{OUTPUT_DIRECTORY_NAME}/{DATASET_NAME}"
RESULTS_PATH = REPO_PATH / "result" / EXPERIMENT_NAME
RESULTS_PATH.mkdir(exist_ok=True, parents=True)
MODELS_PATH = REPO_PATH / "models" / EXPERIMENT_NAME
MODELS_PATH.mkdir(exist_ok=True, parents=True)

NPZ_SAVE_LOC = (
    REPO_PATH
    / "pnet_data/processed_files"
    / OUTPUT_DIRECTORY_NAME
    / DATASET_NAME
    / "SavedNpz"
    / f"deltaR={MAX_DISTANCE}"
    / f"energy_scale={ENERGY_SCALE}"
)

MAX_SAMPLE_LENGTH = 278
BATCH_SIZE = 480
TRAIN_DIR = NPZ_SAVE_LOC / "train"
VAL_DIR = NPZ_SAVE_LOC / "val"


# TODO: move load and data_generator to utils_functs and import from there
def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data["feats"][:, :MAX_SAMPLE_LENGTH, 4:]  # discard tracking information
    frac_labels = data["frac_labels"][:, :MAX_SAMPLE_LENGTH]
    energy_weights = data["tot_truth_e"][:, :MAX_SAMPLE_LENGTH]
    return feats, frac_labels, energy_weights


# works with BS but don't fix last batch size
def data_generator(data_dir, batch_size, drop_last=True):
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if len(npz_files) == 0:
        raise Exception(f"{data_dir} does not contain npz files!")

    np.random.shuffle(npz_files)
    for npz_file in npz_files:
        feats, frac_labels, e_weights = load_data_from_npz(npz_file)
        dataset_size = feats.shape[0]
        for i in range(0, dataset_size, batch_size):
            end_index = i + batch_size
            if end_index > dataset_size:
                if drop_last:
                    continue  # Drop last smaller batch
                else:
                    batch_feats = feats[i:]
                    batch_labels = frac_labels[i:]
                    batch_e_weights = e_weights[i:]
            else:
                batch_feats = feats[i:end_index]
                batch_labels = frac_labels[i:end_index]
                batch_e_weights = e_weights[i:end_index]

            yield (
                batch_feats,
                batch_labels.reshape(*batch_labels.shape, 1),
                batch_e_weights.reshape(*batch_e_weights.shape, 1),
            )


# ============ CHECK LOAD_NPZ ================================================================================


npz_file = glob.glob(os.path.join(TRAIN_DIR, "*.npz"))[0]
print(f"Inspecting file: {npz_file}")
data = np.load(npz_file)
feats = data["feats"]  # [:, :MAX_SAMPLE_LENGTH, 4:]  # discard tracking information
frac_labels = data["frac_labels"]  # [:, :MAX_SAMPLE_LENGTH]
energy_weights = data["tot_truth_e"]  # [:, :MAX_SAMPLE_LENGTH]

print(f"Check shapes:\n{feats.shape=}\n{frac_labels.shape=}\n{energy_weights.shape=}")

WARN_HEADER = " WARNING "
print(
    f"\n{ WARN_HEADER :#^100}\n\nModel `num_features` should be set to {feats.shape[-1]} minus the number of metadata features for data at:\n{Path(npz_file).parent}\n\n{ ' END ' :#^100}"
)

# =======================================================================================================================

# ============ CHECK GENERATOR ================================================================================

# TODO: implement check for different BATCH_SIZE. The idea is to test BATCH_SIZE values above the number of samples in one chunk file (>1/2k should be sufficient)
max_iterations = 10
for drop_last_batch in [True, False]:
    print(f"Checking for {drop_last_batch=}")
    for step, (x_batch_train, y_batch_train, e_weight_train) in enumerate(
        data_generator(TRAIN_DIR, BATCH_SIZE, drop_last=drop_last_batch)
    ):
        if step >= max_iterations:
            break

        assert (
            x_batch_train.shape[0] == BATCH_SIZE
        ), f"X batch dimension do not match: {x_batch_train.shape[0]=} VS {BATCH_SIZE=}"
        assert (
            y_batch_train.shape[0] == BATCH_SIZE
        ), f"Y batch dimension do not match: {y_batch_train.shape[0]=} VS {BATCH_SIZE=}"
        assert (
            e_weight_train.shape[0] == BATCH_SIZE
        ), f"W batch dimension do not match: {e_weight_train.shape[0]=} VS {BATCH_SIZE=}"

        assert (
            x_batch_train.shape[1] == y_batch_train.shape[1]
        ), f"Input and target number of points does not match: { x_batch_train.shape[1]=} VS { y_batch_train.shape[1]=}"
    print("\u2713", f"Done check for {drop_last_batch=}\n\n")


# =======================================================================================================================
